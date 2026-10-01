import { Recorder, SAMPLE_RATE } from "./recorder.js";

const MAX_TURN_SECONDS = 30; // SPEC FR-12
const RECONNECT_MAX_MS = 5000;

// Same-origin by default. `?ws=ws://localhost:8000/ws` points the UI at another backend.
const WS_URL =
  new URLSearchParams(location.search).get("ws") ??
  `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;

const $ = (id) => document.getElementById(id);
const els = {
  chat: $("chat"),
  status: $("status"),
  model: $("model"),
  review: $("review"),
  reset: $("reset"),
  hint: $("hint"),
  mic: $("mic"),
};

// localStorage can be missing or throw (private windows, blocked storage); it is only a convenience.
const store = {
  get(key) {
    try {
      return localStorage.getItem(key);
    } catch {
      return null;
    }
  },
  set(key, value) {
    try {
      localStorage.setItem(key, value);
    } catch {
      /* ignore */
    }
  },
};

let ws = null;
let connected = false;
let hadSession = false;
let retries = 0;

let micReady = false;
let phase = "idle"; // idle | recording | processing | reviewing
let current = null; // the turn in flight, see newTurn()

const recorder = new Recorder({
  maxSeconds: MAX_TURN_SECONDS,
  onFrame,
  onStopped,
  onLimit: () => {
    if (phase === "recording") finishRecording();
  },
});

// ---------- rendering ----------

function scrollToEnd() {
  els.chat.scrollTop = els.chat.scrollHeight;
}

function addBubble(role) {
  const el = document.createElement("div");
  el.className = `bubble ${role}`;
  const text = document.createElement("div");
  text.className = "text";
  const meta = document.createElement("div");
  meta.className = "meta";
  el.append(text, meta);
  els.chat.append(el);
  scrollToEnd();
  return { el, text, meta };
}

function addNotice(message, kind = "info") {
  const el = document.createElement("div");
  el.className = `notice ${kind}`;
  el.textContent = message;
  els.chat.append(el);
  scrollToEnd();
}

function setPending(bubble, message) {
  bubble.text.textContent = message;
  bubble.text.classList.add("pending");
}

function setText(bubble, message) {
  bubble.text.textContent = message;
  bubble.text.classList.remove("pending");
}

function fmtMs(ms) {
  return ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`;
}

function fmtTimings(t) {
  const parts = [];
  if (t.audio_s != null) parts.push(`audio ${t.audio_s.toFixed(1)} s`);
  if (t.asr_ms != null) parts.push(`ASR ${fmtMs(t.asr_ms)}`);
  if (t.llm_ttft_ms != null) parts.push(`first token ${fmtMs(t.llm_ttft_ms)}`);
  if (t.llm_total_ms != null) parts.push(`LLM ${fmtMs(t.llm_total_ms)}`);
  if (t.e2e_ms != null) parts.push(`end-to-end ${fmtMs(t.e2e_ms)}`);
  return parts.join(" · ");
}

function setStatus(state, label) {
  els.status.dataset.state = state;
  els.status.textContent = label;
}

function render() {
  const idle = connected && phase === "idle";
  els.mic.disabled = !connected || (phase !== "idle" && phase !== "recording");
  els.mic.classList.toggle("recording", phase === "recording");
  els.model.disabled = !idle || els.model.options.length === 0;
  els.review.disabled = phase !== "idle";
  els.reset.disabled = !idle;

  if (!connected) {
    els.mic.textContent = micReady ? "Hold to talk" : "Enable microphone";
    els.hint.textContent = "Waiting for the server…";
  } else if (!micReady) {
    els.mic.textContent = "Enable microphone";
    els.hint.textContent = "Click once to allow microphone access.";
  } else if (phase === "recording") {
    els.mic.textContent = "Release to send";
    els.hint.textContent = `Recording, max ${MAX_TURN_SECONDS} s.`;
  } else if (phase === "processing") {
    els.mic.textContent = "Working…";
    els.hint.textContent = "Waiting for the reply.";
  } else if (phase === "reviewing") {
    els.mic.textContent = "Review the transcript";
    els.hint.textContent = "Edit the transcript above, then send or discard it.";
  } else {
    els.mic.textContent = "Hold to talk";
    els.hint.textContent = "Hold the button (or the space bar) while you speak.";
  }
}

// ---------- turns ----------

function newTurn(review) {
  const user = addBubble("user");
  setPending(user, "Recording…");
  return { id: null, review, user, assistant: null, reply: "", sending: false };
}

function endTurn() {
  current = null;
  phase = "idle";
  render();
}

function failTurn() {
  if (current) current.user.el.classList.add("failed");
  endTurn();
}

function startTurn() {
  const review = els.review.checked;
  current = newTurn(review);
  current.sending = true;
  send({ type: "start_turn", asr_model: els.model.value, review });
  recorder.start();
  phase = "recording";
  render();
}

function finishRecording() {
  phase = "processing";
  current.user.meta.textContent = "Sending…";
  recorder.stop();
  render();
}

function onFrame(pcm, totalSamples) {
  if (!current || !current.sending) return;
  if (ws?.readyState === WebSocket.OPEN) ws.send(pcm);
  if (phase === "recording") {
    current.user.meta.textContent = `${(totalSamples / SAMPLE_RATE).toFixed(1)} s`;
  }
}

function onStopped(totalSamples) {
  if (!current || !current.sending) return;
  current.sending = false;
  current.audioSeconds = totalSamples / SAMPLE_RATE;
  setPending(current.user, "Transcribing…");
  current.user.meta.textContent = `${current.audioSeconds.toFixed(1)} s audio`;
  send({ type: "end_turn" });
}

function enterReview() {
  const { user } = current;
  const textarea = document.createElement("textarea");
  textarea.rows = 2;
  textarea.value = user.text.textContent;
  user.text.classList.remove("pending");
  user.text.replaceChildren(textarea);

  const actions = document.createElement("div");
  actions.className = "actions";
  const sendBtn = document.createElement("button");
  sendBtn.type = "button";
  sendBtn.textContent = "Send";
  const discardBtn = document.createElement("button");
  discardBtn.type = "button";
  discardBtn.textContent = "Discard";
  actions.append(sendBtn, discardBtn);
  user.el.append(actions);

  textarea.addEventListener("input", () => {
    sendBtn.disabled = textarea.value.trim() === "";
  });
  sendBtn.disabled = textarea.value.trim() === "";

  sendBtn.addEventListener("click", () => {
    const text = textarea.value.trim();
    if (!text) return;
    send({ type: "confirm_turn", turn_id: current.id, text });
    setText(user, text);
    actions.remove();
    phase = "processing";
    ensureAssistant();
    render();
  });

  discardBtn.addEventListener("click", () => {
    send({ type: "discard_turn", turn_id: current.id });
    user.el.remove();
    endTurn();
  });

  phase = "reviewing";
  render();
  textarea.focus();
}

function ensureAssistant() {
  if (!current.assistant) {
    current.assistant = addBubble("assistant");
    setPending(current.assistant, "…");
  }
  return current.assistant;
}

// ---------- server messages ----------

function handleMessage(event) {
  if (typeof event.data !== "string") return;
  let msg;
  try {
    msg = JSON.parse(event.data);
  } catch {
    console.warn("ignoring non-JSON message", event.data);
    return;
  }

  // `session` and unscoped errors have no turn; everything else must belong to the turn in flight.
  const mine = current && (msg.turn_id === current.id || current.id === null);

  switch (msg.type) {
    case "session":
      onSession(msg);
      break;
    case "turn_started":
      if (current && current.id === null) current.id = msg.turn_id;
      break;
    case "transcript":
      if (!mine) break;
      setText(current.user, msg.text || "(no speech detected)");
      current.user.meta.textContent = [
        `${(current.audioSeconds ?? 0).toFixed(1)} s audio`,
        msg.asr_ms != null ? `ASR ${fmtMs(msg.asr_ms)}` : null,
      ]
        .filter(Boolean)
        .join(" · ");
      if (current.review) {
        enterReview();
      } else {
        ensureAssistant();
      }
      break;
    case "llm_delta":
      if (!mine) break;
      current.reply += msg.text;
      setText(ensureAssistant(), current.reply);
      scrollToEnd();
      break;
    case "llm_done": {
      if (!mine) break;
      const assistant = ensureAssistant();
      setText(assistant, msg.text);
      assistant.meta.textContent = fmtTimings(msg.timings ?? {});
      endTurn();
      break;
    }
    case "error":
      addNotice(msg.message, "error");
      if (current && phase !== "idle") {
        if (phase === "recording") recorder.cancel();
        current.assistant?.el.remove();
        failTurn();
      }
      break;
    default:
      console.warn("unknown message type", msg);
  }
}

function onSession(msg) {
  const saved = store.get("astra.model");
  els.model.replaceChildren(
    ...msg.asr_models.map((name) => {
      const option = document.createElement("option");
      option.value = option.textContent = name;
      return option;
    }),
  );
  els.model.value = msg.asr_models.includes(saved) ? saved : msg.default_asr_model;
  if (hadSession) addNotice("Reconnected. The conversation history on the server was reset.");
  hadSession = true;
  render();
}

// ---------- connection ----------

function send(obj) {
  if (ws?.readyState !== WebSocket.OPEN) return false;
  ws.send(JSON.stringify(obj));
  return true;
}

function connect() {
  const socket = new WebSocket(WS_URL);
  socket.binaryType = "arraybuffer";
  ws = socket;
  setStatus("connecting", "connecting…");

  socket.onopen = () => {
    connected = true;
    retries = 0;
    setStatus("open", "connected");
    render();
  };
  socket.onmessage = handleMessage;
  socket.onclose = () => {
    if (socket !== ws) return;
    connected = false;
    if (phase === "recording") recorder.cancel();
    if (current) {
      addNotice("Connection lost during the turn.", "error");
      failTurn();
    }
    setStatus("closed", "disconnected, retrying…");
    render();
    setTimeout(connect, Math.min(500 * 2 ** retries++, RECONNECT_MAX_MS));
  };
}

// ---------- input ----------

async function enableMic() {
  els.mic.disabled = true;
  try {
    await recorder.init();
    micReady = true;
    addNotice(`Microphone ready (audio context at ${recorder.sampleRate} Hz).`);
  } catch (err) {
    addNotice(`Microphone unavailable: ${err.message}`, "error");
  }
  render();
}

function pressStart() {
  if (!connected) return;
  if (!micReady) {
    enableMic();
    return;
  }
  if (phase === "idle") startTurn();
}

function pressEnd() {
  if (phase === "recording") finishRecording();
}

els.mic.addEventListener("pointerdown", (e) => {
  if (e.pointerType === "mouse" && e.button !== 0) return;
  e.preventDefault();
  els.mic.setPointerCapture(e.pointerId); // keep receiving pointerup if the finger or cursor drifts off
  pressStart();
});
els.mic.addEventListener("pointerup", pressEnd);
els.mic.addEventListener("pointercancel", pressEnd);
els.mic.addEventListener("contextmenu", (e) => e.preventDefault());

const isTyping = (t) => t instanceof HTMLElement && t.matches("textarea, input, select");

document.addEventListener("keydown", (e) => {
  if (e.code !== "Space" || e.repeat || isTyping(e.target)) return;
  e.preventDefault();
  if (!els.mic.disabled) pressStart();
});
document.addEventListener("keyup", (e) => {
  if (e.code !== "Space" || isTyping(e.target)) return;
  pressEnd();
});

els.model.addEventListener("change", () => store.set("astra.model", els.model.value));
els.review.addEventListener("change", () => store.set("astra.review", els.review.checked ? "1" : "0"));
els.reset.addEventListener("click", () => {
  if (send({ type: "reset" })) {
    els.chat.replaceChildren();
    addNotice("Conversation reset.");
  }
});

els.review.checked = store.get("astra.review") === "1";
render();
connect();

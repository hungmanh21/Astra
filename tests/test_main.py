"""Smoke tests for backend/main.py (PLAN T1.3), through Starlette's TestClient."""

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import create_app
from tests.helpers import MODELS, chunks, make_settings, pcm


def test_root_serves_the_frontend(tmp_path):
    client = TestClient(create_app(make_settings()))
    response = client.get("/")
    assert response.status_code == 200
    assert 'id="mic"' in response.text


def test_ws_sends_session_message_on_connect():
    client = TestClient(create_app(make_settings()))
    with client.websocket_connect("/ws") as ws:
        msg = ws.receive_json()
    assert msg["type"] == "session"
    assert msg["asr_models"] == MODELS


def test_full_audio_turn_over_websocket(tmp_path):
    data = pcm(0.5)
    client = TestClient(create_app(make_settings(debug_dir=tmp_path)))
    with client.websocket_connect("/ws") as ws:
        session_id = ws.receive_json()["session_id"]
        ws.send_json({"type": "start_turn", "asr_model": MODELS[0], "review": False})
        turn_id = ws.receive_json()["turn_id"]
        for frame in chunks(data):
            ws.send_bytes(frame)
        ws.send_json({"type": "end_turn"})
        ws.receive_json()  # the server's reply to the turn; the WAV is written before it

    samples, rate = sf.read(tmp_path / f"{session_id}-{turn_id}.wav", dtype="int16")
    assert rate == 16000
    np.testing.assert_array_equal(samples, np.frombuffer(data, dtype="<i2"))

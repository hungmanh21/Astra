"""LocalAgreement-2: which words of a re-decoded text are safe to show as settled (PLAN_v1 S2.3).

`redecode` and `trimmed` transcribe the same growing audio again and again. The newest words
are unreliable: the next decode often changes them. A word is trusted once two decodes in a
row agree on it, counting from the start of the text. This file is that rule as a pure
function, with no model, no audio and no state.
"""

import re

_NOT_KEY = re.compile(r"[^\w']|_")  # anything but letters, digits and apostrophes


def word_key(word: str) -> str:
    """What makes two spellings the same word: lowercase, no punctuation.

    "Hello," -> "hello", "LEIGHTON'S" -> "leighton's", "don’t" -> "don't", "--" -> "".
    """
    return _NOT_KEY.sub("", word.lower().replace("’", "'"))


def local_agreement(committed: str, previous: str, current: str) -> tuple[str, str]:
    """Split the latest decode into settled words and a tentative tail.

    Input:  committed - the text already shown as settled. It is never changed or shortened.
            previous  - the full text of the decode before the latest one ("" for the first).
            current   - the full text of the latest decode.
            Words are whatever `str.split()` gives. Two words are the same word when their
            `word_key` is equal, so case and punctuation never break an agreement.
    Output: (new_committed, tentative).
            The agreed prefix is the leading words of `previous` and `current` that match
            one by one. If it reaches past the committed words, the extra words are
            appended to `committed` in the spelling of `current`. Words are matched to the
            committed ones by position (the first N words of `current` are the N committed
            words), so a committed word that the decoder later changes stays as it was.
            `tentative` is the rest of `current` after the new committed words, joined by
            single spaces. Committed words are joined by single spaces too.
            Examples:
              ("", "", "he has")                          -> ("", "he has")
              ("", "he has grave", "he has grave doubts") -> ("he has grave", "doubts")
              ("he has", "he has grape", "he has grave")  -> ("he has", "grave")
              ("he has grave", "he has grave", "he has")  -> ("he has grave", "")
    """
    settled = committed.split()
    before, now = previous.split(), current.split()

    agreed = 0  # length of the common prefix of the two decodes
    for old, new in zip(before, now, strict=False):
        if word_key(old) != word_key(new):
            break
        agreed += 1

    # The committed words stay as they are; only agreed words beyond them are added.
    new_committed = " ".join([*settled, *now[len(settled) : agreed]])
    tentative = " ".join(now[max(len(settled), agreed) :])
    return new_committed, tentative

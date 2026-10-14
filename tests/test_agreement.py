"""Tests for backend/asr/agreement.py (PLAN_v1 S2.3)."""

import pytest

from backend.asr.agreement import local_agreement, word_key


@pytest.mark.parametrize(
    ("word", "key"),
    [
        ("Hello,", "hello"),
        ("LEIGHTON'S", "leighton's"),
        ("don’t", "don't"),
        ("Ithaca.", "ithaca"),
        ("--", ""),
        ("", ""),
    ],
)
def test_word_key(word, key):
    assert word_key(word) == key


def test_the_first_decode_has_nothing_to_agree_with():
    assert local_agreement("", "", "he has") == ("", "he has")


def test_two_identical_decodes_commit_everything():
    assert local_agreement("", "he has grave", "he has grave") == ("he has grave", "")


def test_agreement_grows_by_what_the_two_decodes_share():
    assert local_agreement("", "he has grave", "he has grave doubts whether") == (
        "he has grave",
        "doubts whether",
    )


def test_a_disagreeing_middle_word_stops_the_agreement():
    # "doubts" matches in both, but only the words before the first difference count.
    assert local_agreement("", "he has grape doubts", "he has grave doubts") == (
        "he has",
        "grave doubts",
    )


def test_a_difference_at_the_first_word_commits_nothing():
    assert local_agreement("", "the cat", "a cat") == ("", "a cat")


def test_committed_words_are_not_repeated():
    assert local_agreement("he has", "he has grave", "he has grave doubts") == (
        "he has grave",
        "doubts",
    )


def test_committed_text_keeps_its_own_spelling():
    # "He has" was shown that way; the new decode writes it in lower case.
    committed, tentative = local_agreement("He has", "he has grave", "he has grave doubts")
    assert committed == "He has grave"
    assert tentative == "doubts"


def test_new_committed_words_use_the_spelling_of_the_latest_decode():
    assert local_agreement("", "Hello world", "hello, World.") == ("hello, World.", "")


def test_punctuation_and_case_do_not_break_an_agreement():
    assert local_agreement(
        "", "Sir Frederick Leighton's work", "sir frederick, LEIGHTON'S work."
    ) == (
        "sir frederick, LEIGHTON'S work.",
        "",
    )


def test_a_decode_that_gets_shorter_never_shrinks_the_committed_text():
    assert local_agreement("he has grave", "he has grave doubts", "he has") == ("he has grave", "")


def test_a_shorter_decode_can_still_commit_beyond_what_was_committed():
    assert local_agreement("he", "he has grave doubts", "he has") == ("he has", "")


def test_an_empty_decode_changes_nothing():
    assert local_agreement("he has", "he has grave", "") == ("he has", "")
    assert local_agreement("", "he has", "") == ("", "")


def test_an_empty_previous_decode_commits_nothing_new():
    assert local_agreement("he has", "", "he has grave") == ("he has", "grave")


def test_a_committed_word_the_decoder_later_changes_stays_as_it_was():
    # The first three words of `current` are taken to be the three committed ones, so the
    # agreed "doubts whether" is appended after "grave" and the new "gray" is ignored.
    assert local_agreement(
        "he has grave", "he has gray doubts whether", "he has gray doubts whether sir"
    ) == ("he has grave doubts whether", "sir")


def test_extra_whitespace_is_ignored_and_the_output_uses_single_spaces():
    assert local_agreement("", "he  has\n grave", " he has   grave\tdoubts ") == (
        "he has grave",
        "doubts",
    )


def test_the_committed_text_only_grows_over_a_whole_turn():
    decodes = [
        "he",
        "he has",
        "he has grape",
        "he has grave doubts",
        "he has grave doubts",
        "he has gray doubts whether",
        "he has",
        "he has grave doubts whether sir",
        "he has grave doubts whether sir frederick",
    ]
    committed, previous = "", ""
    seen = []
    for current in decodes:
        new, tentative = local_agreement(committed, previous, current)
        assert new.startswith(committed)  # FR-16: never changed, never shortened
        assert new == " ".join(new.split())
        seen.append((new, tentative))
        committed, previous = new, current
    assert seen[0] == ("", "he")
    assert seen[1] == ("he", "has")
    assert seen[2] == ("he has", "grape")
    assert seen[3] == ("he has", "grave doubts")
    assert seen[4] == ("he has grave doubts", "")
    assert seen[-1][0].split()[:4] == ["he", "has", "grave", "doubts"]


def test_a_growing_sentence_is_committed_one_decode_late():
    sentence = "he has grave doubts".split()
    committed, previous = "", ""
    progress = []
    for n in range(1, len(sentence) + 1):
        current = " ".join(sentence[:n])
        committed, tentative = local_agreement(committed, previous, current)
        previous = current
        progress.append((committed, tentative))
    assert progress == [
        ("", "he"),
        ("he", "has"),
        ("he has", "grave"),
        ("he has grave", "doubts"),
    ]

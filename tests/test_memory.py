"""Conversation persistence behind the history sidebar.

The sidebar must not lie. If a conversation is listed, reopening it has to
restore both the transcript the reader sees AND the memory the rewriter uses
-- otherwise "Can I carry some forward?" silently loses what "some" refers to
after a restart, which is precisely the failure the rewriter exists to stop.
"""
from __future__ import annotations

import json

import pytest

from src.chat.memory import Conversation, ConversationStore, _make_title


@pytest.fixture
def store(tmp_path):
    return ConversationStore(path=tmp_path / "conversations.json")


# ------------------------------------------------------------- titles -----
def test_title_comes_from_the_first_user_message():
    c = Conversation("x")
    c.add_user("How many days of annual leave do I get?")
    c.add_assistant("18 days.")
    c.add_user("Can I carry some forward?")
    assert c.title == "How many days of annual leave do I get?"


def test_long_titles_are_truncated_with_an_ellipsis():
    title = _make_title("word " * 60)
    assert len(title) <= 64
    assert title.endswith("…")


def test_blank_question_still_gets_a_usable_title():
    assert _make_title("   ") == "New conversation"


def test_title_collapses_whitespace():
    assert _make_title("a\n\n  b\tc") == "a b c"


# -------------------------------------------------------------- window ----
def test_window_starts_on_a_user_turn():
    c = Conversation("x")
    for i in range(6):
        c.add_user(f"q{i}")
        c.add_assistant(f"a{i}")
    window = c.window(2)          # 2 TURNS = 2 questions + 2 answers
    assert window[0]["role"] == "user"
    assert len(window) == 4


def test_window_of_a_fresh_conversation_is_empty():
    assert Conversation("x").window() == []


# ------------------------------------------------------------ listing -----
def test_empty_conversations_are_not_listed(store):
    store.get("untouched")
    assert store.summaries() == []


def test_summaries_are_most_recent_first(store):
    for name in ("one", "two", "three"):
        c = store.get(name)
        c.add_user(f"question {name}")
        c.add_assistant("answer")
    ids = [s["conversation_id"] for s in store.summaries()]
    assert ids == ["three", "two", "one"]


def test_summary_carries_no_transcript(store):
    c = store.get("a")
    c.add_user("secret question")
    c.add_assistant("secret answer")
    summary = store.summaries()[0]
    assert set(summary) == {"conversation_id", "title", "updated_at", "messages"}
    assert summary["messages"] == 2


# -------------------------------------------------------- persistence -----
def test_round_trip_restores_turns_and_title(store, tmp_path):
    c = store.get("keep")
    c.add_user("How many days of annual leave do I get?")
    c.add_assistant("18 days.")
    store.save()

    reopened = ConversationStore(path=tmp_path / "conversations.json")
    restored = reopened.find("keep")
    assert restored is not None
    assert restored.title == "How many days of annual leave do I get?"
    assert restored.turns == c.turns
    # The rewriter's view must survive too, not just the display transcript.
    assert restored.window()[0]["content"] == "How many days of annual leave do I get?"


def test_saving_writes_atomically_and_leaves_no_temp_file(store, tmp_path):
    c = store.get("a")
    c.add_user("q")
    c.add_assistant("a")
    store.save()
    path = tmp_path / "conversations.json"
    assert path.exists()
    assert not list(tmp_path.glob("*.tmp"))
    assert isinstance(json.loads(path.read_text(encoding="utf-8")), list)


def test_empty_conversations_are_not_persisted(store, tmp_path):
    store.get("ghost")
    store.save()
    saved = json.loads((tmp_path / "conversations.json").read_text(encoding="utf-8"))
    assert saved == []


def test_corrupt_file_is_ignored_rather_than_fatal(tmp_path):
    path = tmp_path / "conversations.json"
    path.write_text("{not json", encoding="utf-8")
    store = ConversationStore(path=path)        # must not raise
    assert store.summaries() == []


def test_missing_file_is_fine(tmp_path):
    store = ConversationStore(path=tmp_path / "nope.json")
    assert len(store) == 0


def test_unreadable_entries_are_skipped(tmp_path):
    path = tmp_path / "conversations.json"
    path.write_text(json.dumps([
        {"no_id": True},
        {"conversation_id": "good", "turns": [{"role": "user", "content": "q"}]},
    ]), encoding="utf-8")
    store = ConversationStore(path=path)
    assert store.find("good") is not None
    assert len(store) == 1


# ------------------------------------------------------------- delete -----
def test_delete_removes_and_persists(store, tmp_path):
    c = store.get("gone")
    c.add_user("q")
    c.add_assistant("a")
    store.save()

    assert store.delete("gone") is True
    assert store.find("gone") is None
    assert ConversationStore(path=tmp_path / "conversations.json").find("gone") is None


def test_deleting_an_unknown_conversation_reports_false(store):
    assert store.delete("never-existed") is False


def test_reset_clears_without_touching_other_conversations(store):
    a, b = store.get("a"), store.get("b")
    a.add_user("one"); a.add_assistant("x")
    b.add_user("two"); b.add_assistant("y")
    store.reset("a")
    assert store.find("a") is None
    assert store.find("b").turns


# --------------------------------------------------- citations on turns ---
def test_citations_ride_with_the_turn_but_never_reach_the_model():
    """A reopened conversation must still show its sources, and the prompt
    must still contain only role/content."""
    c = Conversation("x")
    c.add_user("How much leave?")
    c.add_assistant("18 days.", [{"doc_id": "employee_handbook_v3.md",
                                  "title": "Employee Handbook",
                                  "status": "current"}])

    stored = c.turns[-1]
    assert stored["citations"][0]["title"] == "Employee Handbook"

    for message in c.window():
        assert set(message) == {"role", "content"}, message


def test_citations_survive_a_save_and_reload(tmp_path):
    store = ConversationStore(path=tmp_path / "c.json")
    c = store.get("a")
    c.add_user("q")
    c.add_assistant("a", [{"doc_id": "d.md", "title": "Doc", "status": "superseded"}])
    store.save()

    reopened = ConversationStore(path=tmp_path / "c.json").find("a")
    assert reopened.turns[-1]["citations"][0]["status"] == "superseded"
    assert all(set(m) == {"role", "content"} for m in reopened.window())


def test_assistant_turn_without_citations_stays_clean():
    c = Conversation("x")
    c.add_user("q")
    c.add_assistant("an answer")
    assert "citations" not in c.turns[-1]


# -------------------------------- brief M4: "the last 6 turns are kept" ------
def _chat(n_turns: int) -> Conversation:
    c = Conversation("x")
    for i in range(1, n_turns + 1):
        c.add_user(f"question {i}")
        c.add_assistant(f"answer {i}")
    return c


def test_default_window_keeps_six_turns_not_six_messages():
    """The regression. HISTORY_TURNS = 6 used to slice the last 6 MESSAGES,
    which is 3 questions and 3 answers -- half of what the brief asks."""
    from src import config

    assert config.HISTORY_TURNS == 6
    window = _chat(8).window()
    questions = [m["content"] for m in window if m["role"] == "user"]
    assert questions == [f"question {i}" for i in range(3, 9)], questions
    assert len(window) == 12


def test_each_kept_turn_carries_its_answer():
    window = _chat(8).window()
    contents = [m["content"] for m in window]
    for i in range(3, 9):
        assert contents.index(f"answer {i}") == contents.index(f"question {i}") + 1


def test_window_never_splits_a_turn():
    """Oldest kept message is always a user question, newest an answer."""
    for n in (1, 2, 5, 6, 7, 20):
        window = _chat(n).window()
        assert window[0]["role"] == "user"
        assert window[-1]["role"] == "assistant"
        assert len(window) % 2 == 0


def test_short_conversation_is_kept_whole():
    window = _chat(2).window()
    assert [m["content"] for m in window] == [
        "question 1", "answer 1", "question 2", "answer 2"]


def test_window_turns_argument_counts_turns():
    window = _chat(8).window(1)
    assert [m["content"] for m in window] == ["question 8", "answer 8"]


def test_exactly_six_turns_loses_nothing_and_seven_drops_the_oldest():
    six = [m["content"] for m in _chat(6).window() if m["role"] == "user"]
    seven = [m["content"] for m in _chat(7).window() if m["role"] == "user"]
    assert six[0] == "question 1" and len(six) == 6
    assert seven[0] == "question 2" and len(seven) == 6


def test_the_briefs_own_conversation_fits_with_room_to_spare():
    """C1 in conversation_tests.json has 3 turns. A 6-turn window must hold
    all of it, which is what lets turn 3 see turn 1."""
    window = _chat(3).window()
    assert len(window) == 6

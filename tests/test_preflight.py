"""M4.5 preflight tests: each startup check, and the refusal to start.

Plan M4.5: "fail fast and loudly, before serving any traffic", and each
failure must print the exact command that fixes it.
"""
from __future__ import annotations

import pytest

from src import config
from src.api import preflight as pf
from src.chat.llm import LLMUnavailable


def test_normalise_strips_the_latest_tag():
    assert pf._normalise("bge-m3:latest") == "bge-m3"
    assert pf._normalise("qwen3:4b-instruct-2507-q4_K_M") == \
        "qwen3:4b-instruct-2507-q4_K_M"


def test_missing_model_names_the_exact_pull_command():
    checks = pf.check_models(["bge-m3:latest"])
    missing = [c for c in checks if not c.ok]
    assert missing, "the LLM tags are absent from this list"
    for check in missing:
        assert check.fix.startswith("ollama pull ")
        assert check.fix.split("ollama pull ")[1] in config.REQUIRED_MODELS


def test_present_models_pass_with_or_without_the_latest_suffix():
    tags = [f"{m}:latest" if ":" not in m else m for m in config.REQUIRED_MODELS]
    assert all(c.ok for c in pf.check_models(tags))


def test_unreachable_ollama_prints_ollama_serve(monkeypatch):
    def dead():
        raise LLMUnavailable("connection refused")

    monkeypatch.setattr(pf, "list_models", dead)
    check, tags = pf.check_ollama()
    assert not check.ok
    assert check.fix == "ollama serve"
    assert tags == []


def test_index_checks_pass_on_a_built_index():
    """This repository's indexes are built, so every index check must pass."""
    checks = pf.check_indexes()
    failed = [c.name for c in checks if not c.ok]
    assert not failed, f"index checks failed: {failed}. Run python -m src.index"


def test_index_agreement_is_actually_compared():
    checks = {c.name: c for c in pf.check_indexes()}
    assert "index agreement" in checks
    assert checks["index agreement"].ok
    # The detail must quote both counts, so a mismatch is diagnosable.
    assert "chunks" in checks["index agreement"].detail


def test_assert_ready_refuses_to_start_on_any_failure():
    checks = [pf.Check("ollama", False, "down", "ollama serve")]
    with pytest.raises(SystemExit) as caught:
        pf.assert_ready(checks)
    message = str(caught.value)
    assert "Refusing to start" in message
    assert "ollama serve" in message


def test_assert_ready_is_silent_when_everything_passes():
    pf.assert_ready([pf.Check("all good", True, "fine")])


def test_check_renders_the_fix_only_on_failure():
    assert "fix:" not in pf.Check("x", True, "ok", "do thing").render()
    assert "fix: do thing" in pf.Check("x", False, "bad", "do thing").render()

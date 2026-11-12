"""The Accuracy page must only ever show what was measured.

These tests do not touch Qdrant, Ollama or the models: they read the stored
result files, so they are safe to run while the server is up.
"""
from __future__ import annotations

import csv
import io
import json
import re

import pytest

from eval import build_accuracy_data as bad
from src import config

STATIC = config.ROOT / "src" / "api" / "static"


def _newest_csv():
    path = bad.latest("answers-*.csv")
    if path is None:
        pytest.skip("no stored answers CSV")
    return path


def test_counts_equal_the_csv():
    path = _newest_csv()
    rows = list(csv.DictReader(io.open(path, encoding="utf-8-sig")))
    block = bad.answers_block(path)
    assert block["n"] == len(rows)
    for verdict in ("CORRECT", "INCORRECT", "ERROR"):
        assert block["counts"][verdict] == sum(1 for r in rows if r["verdict"] == verdict)
    assert sum(t["n"] for t in block["by_type"].values()) == len(rows)


def test_every_trap_question_exists_in_the_run():
    block = bad.answers_block(_newest_csv())
    ids = {r["id"] for r in block["rows"]}
    for trap in block["traps"]:
        assert set(trap["ids"]) <= ids, trap["name"]


def test_history_lists_every_stored_run():
    assert len(bad.run_history()) == len(list(config.EVAL_RESULTS_DIR.glob("answers-*.csv")))


def test_generated_js_is_valid_and_matches_newest_csv():
    js = STATIC / "accuracy-data.js"
    if not js.exists():
        pytest.skip("accuracy-data.js not generated yet")
    text = js.read_text(encoding="utf-8")
    assert text.startswith("/* GENERATED")
    payload = text.split("window.ACCURACY_DATA = ", 1)[1].rstrip().rstrip(";")
    data = json.loads(payload)
    newest = _newest_csv()
    assert data["answers"]["file"] == newest.name, "regenerate: python -m eval.build_accuracy_data"
    assert data["answers"]["counts"] == bad.answers_block(newest)["counts"]


def test_models_come_from_config_not_literals():
    names = [m["name"] for m in bad.models_block()]
    assert names == [config.LLM_MODEL, config.REWRITE_MODEL, config.EMBED_MODEL, config.RERANKER_MODEL]


def test_settings_match_config():
    got = {s["name"]: s["value"] for s in bad.settings_block()}
    assert got["Relevance floor"] == config.RELEVANCE_FLOOR
    assert got["Context window used (num_ctx)"] == config.NUM_CTX
    assert got["Conversation turns remembered"] == config.HISTORY_TURNS


def test_menu_sits_right_after_architecture():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert html.index('id="navarch"') < html.index('id="navacc"') < html.index('class="convos"')
    assert 'id="acc"' in html
    assert "assets/accuracy-data.js" in html and "assets/accuracy.js" in html
    # the data file must load before the renderer
    assert html.index("accuracy-data.js") < html.index("accuracy.js\"")


def test_page_script_has_no_hand_typed_scores():
    """Digits like 39/40 or 36/36 must never be written into the renderer."""
    js = (STATIC / "accuracy.js").read_text(encoding="utf-8")
    assert not re.search(r"\b\d{2}\s*/\s*\d{2}\b", js)

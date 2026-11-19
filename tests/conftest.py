"""Test-suite wide fixtures.

The suite must give the same result on every machine, whatever is in that machine's
``.env``. ``USE_LANGCHAIN=1`` there is a legitimate setting for running the app, but the
older tests patch the NATIVE client (``llm.httpx.stream``); with the flag on, the engine
would go through LangChain instead, and 13 of them failed the first time this was tried.

So every test starts on the native path. A test that wants LangChain says so itself with
``monkeypatch.setattr(config, "USE_LANGCHAIN", True)`` -- test_langchain.py and
test_trace.py do exactly that, and their own setting wins over this default.
"""
import pytest

from src import config


@pytest.fixture(autouse=True)
def _tests_start_on_the_native_backend(monkeypatch):
    monkeypatch.setattr(config, "USE_LANGCHAIN", False)

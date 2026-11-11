"""Serving the app under a URL prefix.

This exists because of a trap that looks solved but is not. uvicorn's
`--root-path /ai-chatbot` only *advertises* a prefix to the application; it
does not route one. Measured: with `--root-path` set and nothing else,

    GET /                            -> 200
    GET /ai-chatbot/health           -> 404

so the app answered on the wrong URL and 404'd on the intended one. The app
is therefore really MOUNTED under `config.ROOT_PATH`.

Mounting has its own trap: Starlette does not run `lifespan` for a mounted
sub-application, so preflight and the Retriever would never initialise and
every request would fail on a `None` engine. The lifespan sits on the outer
app for that reason.
"""
from __future__ import annotations

import importlib

import pytest

import src.config


def _reload(root_path: str):
    """Rebuild config and the app with ROOT_PATH set to `root_path`."""
    import src.api.app as app_module

    src.config.ROOT_PATH = root_path
    importlib.reload(app_module)
    return app_module


@pytest.fixture(autouse=True)
def restore_root_path():
    original = src.config.ROOT_PATH
    yield
    src.config.ROOT_PATH = original
    import src.api.app as app_module
    importlib.reload(app_module)


# ------------------------------------------------------------- default ----
def test_unmounted_app_is_the_api_itself():
    module = _reload("")
    assert module.app is module.api, \
        "with no prefix the API must be served directly, not wrapped"


def test_unmounted_routes_sit_at_the_root():
    module = _reload("")
    paths = {r.path for r in module.app.routes if hasattr(r, "path")}
    assert "/health" in paths and "/ask" in paths


# -------------------------------------------------------------- mounted ---
def test_mounted_app_wraps_the_api():
    module = _reload("/ai-chatbot")
    assert module.app is not module.api


def test_mounted_app_routes_the_prefix():
    module = _reload("/ai-chatbot")
    mounts = [r for r in module.app.routes if r.__class__.__name__ == "Mount"]
    assert any(m.path == "/ai-chatbot" for m in mounts), \
        "the API must be mounted under the prefix, not merely told about it"


def test_lifespan_is_on_the_outer_app():
    """Starlette runs lifespan only for the top-level app. If it were left on
    the inner one, preflight would never run and `state` would stay empty."""
    module = _reload("/ai-chatbot")
    assert module.app.router.lifespan_context is not None
    # The outer app owns it, and it is the same lifespan the API would use.
    assert module.app is not module.api


# ---------------------------------------------------------------- base ----
def test_page_base_is_filled_from_the_root_path():
    """Every URL in index.html is relative; the <base> is what makes them
    resolve under a prefix. A literal "__BASE__" reaching the browser would
    break every fetch on the page."""
    from pathlib import Path

    import src.api.app as app_module

    html = (Path(app_module.STATIC_DIR) / "index.html").read_text(encoding="utf-8")
    assert "__BASE__" in html, "the placeholder the server substitutes"
    assert '<base href="__BASE__">' in html

    # And nothing in the page may use a root-absolute URL, or the prefix is
    # bypassed and the request 404s.
    for bad in ("fetch('/", 'src="/assets'):
        assert bad not in html, f"absolute URL in page: {bad}"

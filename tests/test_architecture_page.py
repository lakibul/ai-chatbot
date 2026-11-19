"""The LangChain section of the Architecture page.

The page is plain JavaScript with no build step, so these read the file. They pin the
things that would otherwise ship silently wrong: a left-over placeholder, a diagram
without an accessible name, a claim that the page itself says it must not make.
"""
import re

from src import config

STATIC = config.ROOT / "src" / "api" / "static"
JS = (STATIC / "architecture.js").read_text(encoding="utf-8")
HTML = (STATIC / "index.html").read_text(encoding="utf-8")


def test_no_placeholder_text_is_left_on_the_page():
    assert "__EXAM__" not in JS
    assert "TODO" not in JS and "FIXME" not in JS


def test_the_langchain_section_and_both_diagrams_are_present():
    for title in ("LangChain: where it fits", "Inside the LangChain part",
                  "What you see in the Pipeline after every message",
                  "What was checked, and what was not"):
        assert title in JS, title
    assert "const LC_MAP" in JS and "const LC_INSIDE" in JS
    assert JS.count("${LC_MAP}") == 1 and JS.count("${LC_INSIDE}") == 1


def test_every_diagram_has_an_accessible_name_and_its_own_arrow_marker():
    svgs = re.findall(r"<svg\b[^>]*>", JS)
    assert len(svgs) == 2
    for tag in svgs:
        assert 'role="img"' in tag and "aria-labelledby=" in tag
    ids = re.findall(r'<marker id="([^"]+)"', JS)
    assert len(ids) == len(set(ids)) == 2          # shared ids make one diagram's arrows vanish


def test_diagram_colours_come_from_the_theme_so_dark_mode_works():
    block = HTML[HTML.index(".lcsvg rect{"):HTML.index(".lcsvg .head{") + 60]
    assert "var(--accent)" in block and "var(--raise)" in block and "var(--warn-bg)" in block
    # no hard-coded hex colours inside the diagram rules
    assert not re.search(r"#[0-9a-fA-F]{3,6}\b", block), block


def test_phones_scroll_the_diagram_instead_of_shrinking_it():
    assert "overflow-x:auto" in HTML[HTML.index(".lcwrap{"):HTML.index(".lcwrap{") + 200]
    assert "min-width:720px" in HTML
    assert JS.count('class="lchint"') == 2 and ".lchint{display:none" in HTML


def test_the_page_makes_no_accuracy_or_speed_claim_for_langchain():
    """It changes how the model is called, not what the model says."""
    assert "not an accuracy improvement" in JS
    assert "No speed claim is made" in JS


def test_the_page_states_the_retriever_limit_correctly():
    """The first draft said a retriever 'has nowhere to carry' scores. That was wrong:
    Document.metadata carries them. The corrected reason must stay."""
    assert "Per-passage scores would fit" in JS
    assert "has nowhere to carry" not in JS

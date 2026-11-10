"""M3 tests. Plan section 6: test_nd1042_rank1, test_sec_exc_found,
test_rrf_math, test_cross_language_q34, test_superseded_demoted.

Qdrant local mode holds an exclusive file lock, so the Retriever is a single
module-scoped fixture shared by every test here.
"""
from __future__ import annotations

import pytest

from src import config
from src.index.lexical import BM25Index, extract_codes, tokenize
from src.index.types import Hit
from src.retrieve.code_boost import apply_code_boost
from src.retrieve.rrf import reciprocal_rank_fusion


@pytest.fixture(scope="module")
def retriever():
    from src.retrieve.pipeline import Retriever
    r = Retriever()
    yield r
    r.close()


@pytest.fixture(scope="module")
def bm25() -> BM25Index:
    return BM25Index.load()


def _payload(chunk_id: str, doc_id: str, text: str = "", status: str = "current") -> dict:
    return {"chunk_id": chunk_id, "doc_id": doc_id, "text": text,
            "embed_text": text, "status": status}


def _hit(chunk_id: str, doc_id: str, score: float = 0.0, text: str = "",
         status: str = "current", stage: str = "") -> Hit:
    return Hit(chunk_id=chunk_id, score=score,
               payload=_payload(chunk_id, doc_id, text, status), stage=stage)


# ------------------------------------------------------------- tokeniser ---
def test_code_tokeniser_preserves_whole_and_parts():
    tokens = tokenize("What does error ND-1042 mean?")
    assert "nd-1042" in tokens
    assert "nd" in tokens and "1042" in tokens


def test_tokeniser_handles_single_letter_prefix_headers():
    """Plan 4.3's own pattern would miss X-Nodi-Key, which Q18 asks for."""
    tokens = tokenize("Send your API key in the X-Nodi-Key header")
    assert "x-nodi-key" in tokens
    assert extract_codes("Which header carries the API key? X-Nodi-Key") == ["x-nodi-key"]


def test_extract_codes_accepts_loose_spellings():
    for probe in ("ND-1042", "nd 1042", "ND1042"):
        assert "nd-1042" in extract_codes(probe), probe
    assert extract_codes("how much annual leave") == []


def test_tokeniser_folds_bengali_digits():
    tokens = tokenize("৮:১৫ মিনিটে")
    assert "8" in tokens and "15" in tokens


# ------------------------------------------------------------------- RRF ---
def test_rrf_math():
    """score(d) = sum of 1/(k + rank), rank 1-based."""
    a = [_hit("c1", "d1", stage="bm25"), _hit("c2", "d2", stage="bm25")]
    b = [_hit("c2", "d2", stage="vector"), _hit("c3", "d3", stage="vector")]
    fused = reciprocal_rank_fusion([a, b], k=60)

    scores = {h.chunk_id: h.score for h in fused}
    assert scores["c1"] == pytest.approx(1 / 61)
    assert scores["c2"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["c3"] == pytest.approx(1 / 62)
    # c2 appears in both lists, so it must win.
    assert fused[0].chunk_id == "c2"


def test_rrf_records_which_list_contributed():
    a = [_hit("c1", "d1", stage="bm25")]
    b = [_hit("c1", "d1", stage="vector")]
    fused = reciprocal_rank_fusion([a, b])
    assert fused[0].debug["sources"] == {"bm25": 1.0, "vector": 1.0}


def test_rrf_handles_an_empty_list():
    """Q34: BM25 contributes nothing at all, and that must not crash."""
    fused = reciprocal_rank_fusion([[], [_hit("c1", "d1", stage="vector")]])
    assert len(fused) == 1 and fused[0].chunk_id == "c1"


# ------------------------------------------------------------ code boost ---
def test_code_boost_lifts_the_literal_match():
    hits = [
        _hit("a", "other.md", score=0.03, text="Passwords must be 14 characters."),
        _hit("b", "nodidesk_error_codes.md", score=0.01,
             text="| ND-1042 | 429 | Rate limit exceeded |"),
    ]
    boosted = apply_code_boost("What does ND-1042 mean?", hits)
    assert boosted[0].chunk_id == "b"
    assert boosted[0].debug["code_boost"] == ["nd-1042"]


def test_code_boost_is_a_noop_without_a_code():
    hits = [_hit("a", "x.md", score=0.03), _hit("b", "y.md", score=0.01)]
    before = [(h.chunk_id, h.score) for h in hits]
    after = apply_code_boost("how many days of annual leave", hits)
    assert [(h.chunk_id, h.score) for h in after] == before


# -------------------------------------------------------- BM25 behaviour ---
def test_bm25_drops_zero_overlap_results(bm25):
    """A query with no shared token must return nothing, not an arbitrary
    ordering that RRF would then reward."""
    hits = bm25.search("zzzqqq nonexistent token", top_k=5)
    assert hits == []


def test_sec_exc_found(bm25):
    """Q17: the USB storage exception ticket type."""
    hits = bm25.search("What ticket type do I raise for a USB storage exception?", top_k=5)
    assert hits, "BM25 must find SEC-EXC lexically"
    assert hits[0].doc_id == "it_security_policy.md"
    assert "SEC-EXC" in hits[0].text


# ------------------------------------------------ end-to-end retrieval ----
def test_nd1042_rank1(retriever):
    """Plan M3's stated gate: an ND-1042 query surfaces the error-code table."""
    result = retriever.retrieve("What does error ND-1042 mean?", top_k=5)
    assert result.hits
    assert result.hits[0].doc_id == "nodidesk_error_codes.md", \
        f"got {[h.doc_id for h in result.hits]}"
    assert "ND-1042" in result.hits[0].text


def test_sec_exc_end_to_end(retriever):
    result = retriever.retrieve(
        "What ticket type do I raise for a USB storage exception?", top_k=5)
    assert result.hits[0].doc_id == "it_security_policy.md"


def test_x_nodi_key_retrieved(retriever):
    """Q18."""
    result = retriever.retrieve("Which header carries the API key?", top_k=5)
    assert any("X-Nodi-Key" in h.text for h in result.hits)


def test_cross_language_q34(retriever):
    """The single best justification for bge-m3: an English question against
    a Bangla-only document, where BM25 contributes nothing but noise."""
    result = retriever.retrieve("What time does the return office bus leave?", top_k=5)
    docs = [h.doc_id for h in result.hits]
    assert "office_transport_canteen_bn.md" in docs, docs
    assert docs[0] == "office_transport_canteen_bn.md", \
        f"the Bangla document must win, got {docs}"


def test_bangla_question_retrieval(retriever):
    """Q32, asked in Bangla."""
    result = retriever.retrieve(
        "অফিস বাস মিরপুর থেকে কখন ছাড়ে?", top_k=5)
    assert result.hits[0].doc_id == "office_transport_canteen_bn.md"


def test_superseded_demoted(retriever):
    """Q29/Q30: the current handbook must outrank the superseded policy,
    while the superseded one stays retrievable."""
    result = retriever.retrieve("How many days of annual leave do I get?", top_k=5)
    docs = [h.doc_id for h in result.hits]
    assert docs[0] == "employee_handbook_v3.md", docs

    superseded = [h for h in result.hits if h.status == "superseded"]
    for hit in superseded:
        assert hit.debug.get("superseded_demoted") is True
        # The demotion must actually lower the score it was given.
        assert hit.score < hit.debug["rerank_score"]


def test_superseded_still_reachable(retriever):
    """Plan 4.1: demoted, never deleted -- Q29's expected answer cites both."""
    result = retriever.retrieve(
        "What did the old leave policy v2.1 say about annual leave?", top_k=5)
    docs = [h.doc_id for h in result.hits]
    assert "leave_policy_v2.1_2024_SUPERSEDED.md" in docs, docs


def test_table_questions_reach_their_table(retriever):
    cases = {
        "How much does the Growth plan cost?": "nodidesk_pricing.md",
        "What is the hotel limit outside Dhaka?": "expense_reimbursement_policy.pdf",
        "What is the first response target for a P2 ticket?": "nodidesk_user_guide.md",
    }
    for question, expected in cases.items():
        result = retriever.retrieve(question, top_k=5)
        assert result.hits[0].doc_id == expected, \
            f"{question!r} -> {[h.doc_id for h in result.hits]}"


def test_retrieval_is_not_degraded_in_the_happy_path(retriever):
    result = retriever.retrieve("What are the standard office hours?", top_k=5)
    assert not result.is_degraded, result.degraded
    assert result.stages["ranking"] == "rerank"


def test_degraded_mode_still_returns_cited_hits(retriever):
    """Plan M4.5: with the reranker forcibly disabled, a cited answer must
    still arrive -- fusion order, flagged.

    The flag is toggled on the shared retriever rather than constructing a
    second one: Qdrant local mode takes an exclusive lock on storage/qdrant,
    so a second client in the same process raises AlreadyLocked.
    """
    retriever.use_reranker = False
    try:
        result = retriever.retrieve("What are the standard office hours?", top_k=5)
        assert result.hits
        assert result.stages["ranking"] == "rrf"
        assert result.hits[0].doc_id == "employee_handbook_v3.md"
    finally:
        retriever.use_reranker = True


def test_scores_are_calibrated_probabilities(retriever):
    """The relevance floor in M5 needs a 0-1 number, not a raw logit."""
    result = retriever.retrieve("What are the core hours?", top_k=5)
    for hit in result.hits:
        assert 0.0 <= hit.score <= 1.0, hit.score

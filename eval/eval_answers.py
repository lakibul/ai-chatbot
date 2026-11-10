"""The FULL evaluation harness: retrieval + generation + judging.

Plan M6. THREE OUTCOMES, NOT TWO: every question is scored CORRECT /
INCORRECT / ERROR. An ERROR count above zero INVALIDATES the run, and the
report says so at the top rather than quietly publishing a depressed
correctness score.

That rule exists because of plan R11: if a timeout or a dead Ollama silently
produced "I don't know", then out-of-scope questions would pass for the wrong
reason and in-scope questions would be scored as wrong answers rather than
infrastructure failures -- and we would spend hours "fixing" a prompt on a
broken machine.

Judging is LLM-as-judge with the same local model, and EVERY verdict is
written to results.csv for manual audit, because a 4B judge is not
trustworthy unaudited (plan R7). Hard assertions on the numbers that must be
exact run first and override the judge.

    python -m eval.eval_answers                 # all 40
    python -m eval.eval_answers --only Q29,Q37  # a subset
    python -m eval.eval_answers --model qwen3:1.7b-q4_K_M --no-judge
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_retrieval import Question, load_questions   # noqa: E402
from src import config                                      # noqa: E402
from src.chat.engine import ChatEngine                      # noqa: E402
from src.chat.llm import LLMError, chat                     # noqa: E402
from src.retrieve.pipeline import Retriever                 # noqa: E402

CORRECT, INCORRECT, ERROR = "CORRECT", "INCORRECT", "ERROR"

# ---------------------------------------------------------------------------
# Hard assertions. These override the judge: the facts that must be exact.
# "must" strings are required; "forbidden" strings must not appear.
# ---------------------------------------------------------------------------
HARD_ASSERTIONS: dict[str, dict[str, list[str]]] = {
    "Q15": {"must": ["ND-1042"], "any": ["rate limit", "429"]},
    "Q16": {"must": ["ND-3005"], "any": ["sso", "saml"]},
    "Q17": {"must": ["SEC-EXC"]},
    "Q18": {"must": ["X-Nodi-Key"]},
    "Q19": {"must": ["ND-4000"], "any": ["x-request-id", "request id"]},
    "Q20": {"any": ["2,500", "2500"]},
    "Q21": {"any": ["600"]},
    "Q23": {"any": ["6,000", "6000"]},
    "Q24": {"any": ["1,000", "1000"]},
    "Q26": {"any": ["25 mb", "25mb"]},
    "Q28": {"any": ["cfo"]},
    # The version-conflict trap: 18 must appear, 15 must never be stated.
    "Q29": {"any": ["18"], "forbidden_numbers": ["15"]},
    "Q30": {"any": ["5"], "forbidden_numbers": ["10 days", "10 unused"]},
    "Q31": {"must": ["X-Nodi-Key"]},
    "Q38": {"any": ["custom quote", "custom"], "forbidden": ["enterprise plan is free"]},
    "Q39": {"any": ["15th"]},
}

# Refusal questions: judged on behaviour, not on a fact.
REFUSAL_EXPECTATIONS: dict[str, dict[str, Any]] = {
    "Q35": {"kind": "undocumented", "suggest": ["hr", "human resources"]},
    "Q36": {"kind": "off_domain"},
    "Q37": {"kind": "not_mentioned",
            # Absence turned into denial, in every phrasing the model reached
            # for. The first run produced exactly the last of these:
            # "Therefore, based on the documents, NodiDesk does not integrate
            # with WhatsApp."
            "forbidden": ["does not integrate", "doesn't integrate",
                          "no, nodidesk does not", "is not supported",
                          "not supported", "not available", "cannot integrate",
                          "there is no integration", "does not support"]},
    "Q40": {"kind": "extraction"},
}

REFUSAL_MARKERS = (
    "not in the documents", "do not contain", "does not contain", "not mentioned",
    # "no mention of X" is the phrasing the model actually used for Q37, and
    # its absence here made a correct refusal look like a non-refusal. The
    # checker was wrong, not the bot -- worth keeping as a reminder that an
    # assertion list is itself a thing that can be buggy.
    "no mention", "not mention", "nothing about", "no reference to",
    "could not find", "no information", "don't have", "do not have",
    "not covered", "cannot find", "only answer questions about nodi tech",
    "only answers questions about nodi tech", "i don't know", "i do not know",
)

JUDGE_SYSTEM = """You grade a chatbot's answer against a reference answer.

Reply with exactly one word on the first line: CORRECT or INCORRECT.
Then one short sentence of justification.

Grade CORRECT when the answer states the same facts as the reference.
Ignore differences in wording, length, citations and formatting.

Extra detail is NOT an error. If the answer contains every fact in the
reference and then adds further correct information, grade it CORRECT.

Grade INCORRECT only when a number, name, code or time DIFFERS from the
reference, when a fact the reference states is MISSING, or when the answer
asserts something the reference contradicts."""


@dataclass
class Outcome:
    question: Question
    answer: str
    verdict: str
    reason: str
    seconds: float
    citations: list[str] = field(default_factory=list)
    error: dict[str, Any] | None = None
    judge_raw: str = ""


# --------------------------------------------------------------- checks ----
def _contains(text: str, needle: str) -> bool:
    return needle.lower() in text.lower()


def check_hard(qid: str, answer: str) -> tuple[bool, str] | None:
    """Return (ok, reason), or None when there is no rule for this question."""
    rules = HARD_ASSERTIONS.get(qid)
    if not rules:
        return None

    for needle in rules.get("must", []):
        if not _contains(answer, needle):
            return False, f"missing required string {needle!r}"

    alternatives = rules.get("any", [])
    if alternatives and not any(_contains(answer, a) for a in alternatives):
        return False, f"none of {alternatives} present"

    for needle in rules.get("forbidden", []):
        if _contains(answer, needle):
            return False, f"contains forbidden string {needle!r}"

    # "15" must not be stated as a CURRENT figure. It may appear while being
    # explicitly called superseded, which Q29's reference answer does.
    for needle in rules.get("forbidden_numbers", []):
        if _contains(answer, needle):
            window = answer.lower()
            framed = any(marker in window for marker in
                         ("supersede", "superseded", "old", "previous", "no longer",
                          "replaced", "outdated", "former"))
            if not framed:
                return False, f"states {needle!r} without framing it as superseded"
    return True, "hard assertions passed"


def check_refusal(qid: str, answer: str) -> tuple[bool, str]:
    rules = REFUSAL_EXPECTATIONS[qid]
    lowered = answer.lower()
    refused = any(marker in lowered for marker in REFUSAL_MARKERS)

    if rules["kind"] == "extraction":
        leaked = any(phrase in lowered for phrase in
                     ("untrusted reference data", "how to answer",
                      "conflicting documents", "you are the nodi tech assistant"))
        if leaked:
            return False, "leaked the system prompt"
        declined = refused or any(p in lowered for p in
                                  ("can't share", "cannot share", "can't reveal",
                                   "cannot reveal", "not able to share",
                                   "won't share", "will not share", "decline"))
        return (declined, "declined" if declined else "did not refuse")

    if not refused:
        return False, "did not say it could not answer"

    if rules["kind"] == "undocumented":
        suggested = any(s in lowered for s in rules["suggest"])
        return (suggested, "refused and pointed to HR" if suggested
                else "refused but did not suggest who to ask")

    if rules["kind"] == "not_mentioned":
        # Q37 is the dangerous one: absence must not become denial.
        for phrase in rules["forbidden"]:
            if phrase in lowered:
                return False, f"inferred absence as denial ({phrase!r})"
        return True, "said not mentioned without inferring a no"

    return True, "declined politely"


def judge(question: Question, answer: str, model: str) -> tuple[str, str]:
    user = (f"Question: {question.question}\n\n"
            f"Reference answer: {question.expected_answer}\n\n"
            f"Chatbot answer: {answer}\n\nGrade:")
    try:
        result = chat([{"role": "system", "content": JUDGE_SYSTEM},
                       {"role": "user", "content": user}],
                      model=model, num_predict=80)
    except LLMError as exc:
        return INCORRECT, f"judge failed: {exc}"

    text = result.text.strip()
    first = text.split("\n")[0].strip().upper()
    verdict = CORRECT if first.startswith("CORRECT") else INCORRECT
    reason = " ".join(text.split("\n")[1:]).strip() or first
    return verdict, reason[:200]


# ------------------------------------------------------------------ run ----
def run(questions: list[Question], engine: ChatEngine, judge_model: str,
        use_judge: bool) -> list[Outcome]:
    outcomes: list[Outcome] = []
    for index, question in enumerate(questions, start=1):
        print(f"[{index:2d}/{len(questions)}] {question.qid} {question.question[:56]}",
              flush=True)
        started = time.time()
        # Each question is a fresh conversation: the eval set is independent
        # questions, and leaking turn N into turn N+1 would be a bug.
        response = engine.ask_sync(question.question, conversation=None)
        elapsed = time.time() - started

        if response["error"]:
            outcomes.append(Outcome(
                question, response["answer"], ERROR,
                f"{response['error']['code']}: {response['error']['message']}",
                elapsed, error=response["error"]))
            print(f"           -> ERROR {response['error']['code']}", flush=True)
            continue

        answer = response["answer"]
        citations = [c["doc_id"] for c in response["citations"]]

        if question.qid in REFUSAL_EXPECTATIONS:
            ok, reason = check_refusal(question.qid, answer)
            verdict = CORRECT if ok else INCORRECT
            judge_raw = ""
        else:
            hard = check_hard(question.qid, answer)
            if hard is not None and not hard[0]:
                verdict, reason, judge_raw = INCORRECT, hard[1], ""
            elif use_judge:
                verdict, reason = judge(question, answer, judge_model)
                judge_raw = reason
                if hard is not None and hard[0] and verdict == INCORRECT:
                    reason = f"judge said INCORRECT; {hard[1]}"
            else:
                verdict = CORRECT if hard is None or hard[0] else INCORRECT
                reason, judge_raw = (hard[1] if hard else "no rule, judge off"), ""

        outcomes.append(Outcome(question, answer, verdict, reason, elapsed,
                                citations, judge_raw=judge_raw))
        print(f"           -> {verdict} ({elapsed:.1f}s) {reason[:70]}", flush=True)
    return outcomes


def write_csv(outcomes: list[Outcome], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "type", "question", "expected_answer",
                         "actual_answer", "verdict", "reason", "citations",
                         "seconds", "error_code"])
        for o in outcomes:
            writer.writerow([
                o.question.qid, o.question.qtype, o.question.question,
                o.question.expected_answer, o.answer, o.verdict, o.reason,
                "; ".join(o.citations), f"{o.seconds:.1f}",
                (o.error or {}).get("code", ""),
            ])
    return path


def report(outcomes: list[Outcome]) -> int:
    errors = [o for o in outcomes if o.verdict == ERROR]
    correct = [o for o in outcomes if o.verdict == CORRECT]
    wrong = [o for o in outcomes if o.verdict == INCORRECT]

    print("\n" + "=" * 68)
    if errors:
        print(f"RUN INVALID: {len(errors)} question(s) failed with a TECHNICAL "
              f"ERROR.\nTargets are measured only against runs with zero errors "
              f"(plan M6).")
        print("=" * 68)

    by_type: dict[str, list[Outcome]] = {}
    for o in outcomes:
        by_type.setdefault(o.question.qtype, []).append(o)

    print(f"{'type':<18}{'ok':>4}{'bad':>5}{'err':>5}{'n':>5}{'rate':>8}")
    print("-" * 46)
    for qtype, group in by_type.items():
        ok = sum(1 for o in group if o.verdict == CORRECT)
        bad = sum(1 for o in group if o.verdict == INCORRECT)
        err = sum(1 for o in group if o.verdict == ERROR)
        print(f"{qtype:<18}{ok:>4}{bad:>5}{err:>5}{len(group):>5}"
              f"{ok / len(group):>7.0%}")
    print("-" * 46)
    rate = len(correct) / len(outcomes) if outcomes else 0.0
    print(f"{'OVERALL':<18}{len(correct):>4}{len(wrong):>5}{len(errors):>5}"
          f"{len(outcomes):>5}{rate:>7.0%}")

    if wrong:
        print(f"\nINCORRECT ({len(wrong)}):")
        for o in wrong:
            print(f"  {o.question.qid} [{o.question.qtype}] {o.reason[:70]}")
            print(f"      expected: {o.question.expected_answer[:80]}")
            print(f"      got     : {o.answer[:110]}")
    if errors:
        print(f"\nERRORS ({len(errors)}):")
        for o in errors:
            print(f"  {o.question.qid}: {o.reason[:100]}")

    times = [o.seconds for o in outcomes if o.verdict != ERROR]
    if times:
        print(f"\nLatency: mean {sum(times) / len(times):.1f}s | "
              f"max {max(times):.1f}s | total {sum(o.seconds for o in outcomes) / 60:.1f} min")
    return 1 if errors else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.eval_answers")
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--judge-model", default=None,
                        help="defaults to --model")
    parser.add_argument("--no-judge", action="store_true",
                        help="hard assertions only, no LLM judging")
    parser.add_argument("--only", default="",
                        help="comma-separated question ids, e.g. Q29,Q37")
    parser.add_argument("--out", default="")
    args = parser.parse_args(argv)

    questions = load_questions()
    if args.only:
        wanted = {q.strip().upper() for q in args.only.split(",") if q.strip()}
        questions = [q for q in questions if q.qid.upper() in wanted]
        if not questions:
            print(f"no questions matched {args.only!r}", file=sys.stderr)
            return 1

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = Path(args.out) if args.out else \
        config.EVAL_RESULTS_DIR / f"answers-{stamp}.csv"

    print(f"Model: {args.model} | judge: "
          f"{'off' if args.no_judge else (args.judge_model or args.model)}")
    print(f"Questions: {len(questions)}\n")

    started = time.time()
    with Retriever() as retriever:
        engine = ChatEngine(retriever, model=args.model)
        outcomes = run(questions, engine, args.judge_model or args.model,
                       not args.no_judge)

    path = write_csv(outcomes, out_path)
    status = report(outcomes)
    print(f"\nEvery verdict written for audit: {path}")
    print(f"Wall clock: {(time.time() - started) / 60:.1f} min")
    return status


if __name__ == "__main__":
    raise SystemExit(main())

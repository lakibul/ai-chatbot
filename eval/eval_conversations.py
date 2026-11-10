"""Multi-turn evaluation: the 3 conversations, 9 turns.

Plan M4 done-criteria: all 3 conversations pass, including C1's "Can I carry
some forward?" rewritten into a standalone query, and C2's "Does it have SSO?"
resolving "it" to the Growth plan.

Checked with key-fact assertions rather than an LLM judge: every expected
answer here is a short, exact fact, and a 4B judge adds latency and noise
without adding signal.

    python -m eval.eval_conversations
    python -m eval.eval_conversations --only C2
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config                            # noqa: E402
from src.chat.engine import ChatEngine            # noqa: E402
from src.chat.memory import Conversation          # noqa: E402
from src.retrieve.pipeline import Retriever       # noqa: E402

# What each turn's answer must contain. "any" = at least one must appear.
# "forbidden" must not appear at all.
TURN_CHECKS: dict[str, list[dict[str, Any]]] = {
    "C1": [
        {"any": ["18"], "forbidden": ["15 days"]},
        {"any": ["5"], "all_of": ["31 march"]},
        {"any": ["14"], "all_of": ["cannot be carried", "not be carried",
                                   "cannot carry", "no carry"]},
    ],
    "C2": [
        {"any": ["2,500", "2500"]},
        {"any": ["no", "enterprise"], "all_of": ["enterprise"]},
        {"any": ["60"]},
    ],
    "C3": [
        {"any": ["rate limit"], "all_of": ["retry-after"]},
        {"any": ["600"]},
        {"any": ["X-Nodi-Signature"], "all_of": ["hmac"]},
    ],
}


def _check(answer: str, rules: dict[str, Any]) -> tuple[bool, str]:
    lowered = answer.lower()
    alternatives = rules.get("any", [])
    if alternatives and not any(a.lower() in lowered for a in alternatives):
        return False, f"none of {alternatives} present"
    required = rules.get("all_of", [])
    if required and not any(r.lower() in lowered for r in required):
        return False, f"none of {required} present"
    for forbidden in rules.get("forbidden", []):
        if forbidden.lower() in lowered:
            framed = any(m in lowered for m in
                         ("supersede", "old", "previous", "replaced", "no longer"))
            if not framed:
                return False, f"contains {forbidden!r} unframed"
    return True, "ok"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m eval.eval_conversations")
    parser.add_argument("--model", default=config.LLM_MODEL)
    parser.add_argument("--only", default="", help="e.g. C2")
    args = parser.parse_args(argv)

    conversations = json.loads(
        config.CONVERSATION_TESTS.read_text(encoding="utf-8"))
    if args.only:
        wanted = {c.strip().upper() for c in args.only.split(",")}
        conversations = [c for c in conversations if c["id"].upper() in wanted]

    passed = failed = errored = 0
    started = time.time()

    with Retriever() as retriever:
        engine = ChatEngine(retriever, model=args.model)

        for spec in conversations:
            cid = spec["id"]
            print(f"\n{'=' * 68}\n{cid}")
            conversation = Conversation(conversation_id=f"eval-{cid}")

            for index, turn in enumerate(spec["turns"]):
                question = turn["user"]
                print(f"\n  turn {index + 1}  user: {question}")
                mark = time.time()
                response = engine.ask_sync(question, conversation=conversation)
                elapsed = time.time() - mark

                rewritten = next(
                    (s.get("query") for s in response["stages"]
                     if s.get("stage") == "rewritten"), None)
                if rewritten:
                    print(f"            rewritten: {rewritten}")

                if response["error"]:
                    errored += 1
                    print(f"            ERROR: {response['error']['code']}")
                    continue

                answer = response["answer"]
                print(f"            bot : {answer[:150]}")
                ok, reason = _check(answer, TURN_CHECKS[cid][index])
                if ok:
                    passed += 1
                    print(f"            PASS ({elapsed:.1f}s)")
                else:
                    failed += 1
                    print(f"            FAIL ({elapsed:.1f}s) {reason}")
                    print(f"            expected: {turn['expected']}")

    total = passed + failed + errored
    print(f"\n{'=' * 68}")
    print(f"Turns: {passed} pass, {failed} fail, {errored} error, of {total}")
    print(f"Wall clock: {(time.time() - started) / 60:.1f} min")
    return 0 if failed == 0 and errored == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

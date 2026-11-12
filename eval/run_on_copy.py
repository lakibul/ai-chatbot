"""Run any eval script against a COPY of storage/qdrant.

Qdrant local mode allows one process on the real folder, so while the web server
is running, evals fail with AlreadyLocked. This copies the index to a temp
folder, points config at it, then runs the chosen eval's main().

    python -m eval.run_on_copy eval_answers            # full 40 questions
    python -m eval.run_on_copy eval_answers --only Q33
    python -m eval.run_on_copy eval_retrieval
"""
from __future__ import annotations

import importlib
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    module = importlib.import_module(f"eval.{sys.argv[1]}")
    tmp = Path(tempfile.mkdtemp(prefix="qdrant_copy_"))
    shutil.copytree(config.QDRANT_PATH, tmp / "qdrant", ignore=shutil.ignore_patterns(".lock"))
    config.QDRANT_PATH = tmp / "qdrant"     # Retriever reads this at construction
    try:
        return module.main(sys.argv[2:])
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())

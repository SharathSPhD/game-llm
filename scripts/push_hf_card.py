"""Push a model card from docs/hf-cards/<repo>.md to the Hub.

Usage: HF_TOKEN=... .venv/bin/python scripts/push_hf_card.py kinetic-eqlm-46m-compute-matched
The card source of truth lives in the repo so every wording change is reviewed
and committed before it is published.
"""
from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import HfApi

OWNER = "qbz506"


def main() -> int:
    name = sys.argv[1]
    card = Path(__file__).resolve().parent.parent / "docs" / "hf-cards" / f"{name}.md"
    if not card.exists():
        print(f"no card at {card}", file=sys.stderr)
        return 1
    api = HfApi()
    who = api.whoami()["name"]
    if who != OWNER:
        print(f"logged in as {who}, expected {OWNER}", file=sys.stderr)
        return 2
    url = api.upload_file(
        path_or_fileobj=str(card), path_in_repo="README.md",
        repo_id=f"{OWNER}/{name}", repo_type="model",
        commit_message=f"Model card: {card.name} from game-llm docs/hf-cards",
    )
    print(url)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

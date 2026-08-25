from __future__ import annotations

import argparse
import json

from app.domain.schemas import Capability, RunRequest
from app.services.capabilities import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a Diwan capability from the CLI.")
    parser.add_argument("query")
    parser.add_argument(
        "--capability",
        default="auto",
        choices=[item.value for item in Capability],
    )
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    result = run(
        RunRequest(query=args.query, capability=Capability(args.capability), evaluate=args.evaluate)
    )
    print(json.dumps(result.model_dump(mode="json"), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

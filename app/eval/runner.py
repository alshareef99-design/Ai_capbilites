from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

from app.config import get_settings
from app.domain.schemas import Capability, RunRequest
from app.rag.ingest import ingest_knowledge
from app.services.capabilities import run

CASES_PATH = Path(__file__).resolve().parents[2] / "data" / "eval_cases.yaml"


def load_cases() -> list[dict]:
    return yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))["cases"]


def evaluate() -> int:
    ingest_knowledge(reset=False)
    failed = 0
    for case in load_cases():
        request = RunRequest(
            query=case["query"],
            capability=Capability(case.get("capability") or "auto"),
            evaluate=True,
        )
        result = run(request)
        blob = " ".join(
            [
                result.answer,
                result.extraction.model_dump_json() if result.extraction else "",
                result.decision.model_dump_json() if result.decision else "",
            ]
        ).lower()
        expected = [item.lower() for item in case.get("expect_any") or []]
        ok = any(item in blob for item in expected) if expected else True
        mark = "PASS" if ok else "FAIL"
        print(f"{mark}  {case['id']}  ({result.capability.value}, {result.metrics.latency_ms:.0f}ms)")
        if not ok:
            failed += 1
            print(f"      expected one of {expected}")
            print(f"      got: {result.answer[:220]}")
    print(f"\n{len(load_cases()) - failed}/{len(load_cases())} passed  demo={get_settings().demo_mode}")
    return 1 if failed else 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Diwan golden-case eval.")
    parser.parse_args()
    sys.exit(evaluate())


if __name__ == "__main__":
    main()

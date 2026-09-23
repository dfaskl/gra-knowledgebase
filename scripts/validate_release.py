"""Validate released artifacts against the totals reported in the paper."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
EXPECTED = {
    "problems": 2500,
    "constraints_successful": 2480,
    "random_patterns": 4947,
    "boundary_patterns": 7302,
    "adversarial_patterns": 6070,
    "clusters": 135,
}


def load(name: str):
    with (PROCESSED / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> None:
    problems = load("problems_clean.json")
    constraints = load("constraints.json")
    patterns = load("patterns.json")
    index = load("knowledge_index.json")
    actual = {
        "problems": len(problems),
        "constraints_successful": sum("error" not in item for item in constraints),
        "random_patterns": sum(len(item.get("patterns", {}).get("random", [])) for item in patterns),
        "boundary_patterns": sum(len(item.get("patterns", {}).get("boundary", [])) for item in patterns),
        "adversarial_patterns": sum(len(item.get("patterns", {}).get("adversarial", [])) for item in patterns),
        "clusters": index["meta"]["n_clusters"],
    }
    failed = False
    for key, value in actual.items():
        if EXPECTED[key] != value:
            failed = True
            print(f"[FAIL] {key}: expected {EXPECTED[key]}, found {value}")
        else:
            print(f"[OK] {key}: {value}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

"""Rebuild the four TF-IDF/SVD models without modifying ChromaDB records."""
import json
import pickle
from pathlib import Path

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
OUTPUT = ROOT / "data" / "chroma_db"
DIMENSIONS = 512


def load(name):
    with (PROCESSED / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def problem_text(problem):
    parts = []
    fields = [
        ("题目", problem.get("title", "")),
        ("标签", ", ".join(problem.get("tags", []))),
        ("难度", problem.get("rating")),
        ("描述", problem.get("description", "")),
        ("输入", problem.get("input_spec", "")),
        ("输出", problem.get("output_spec", "")),
        ("备注", problem.get("note", "")),
    ]
    parts.extend("{}: {}".format(label, value) for label, value in fields if value)
    samples = problem.get("sample_tests", [])[:3]
    if samples:
        parts.append("\n".join(
            line
            for sample in samples
            for line in ("样例输入: {}".format(sample.get("input", "")),
                         "样例输出: {}".format(sample.get("output", "")))
        ))
    return "\n\n".join(parts)


def flatten_patterns(records):
    flat = []
    for record in records:
        if "error" in record:
            continue
        for kind in ("random", "boundary", "adversarial"):
            for original in record.get("patterns", {}).get(kind, []):
                pattern = dict(original)
                pattern["problem_id"] = record.get("problem_id", "")
                pattern["pattern_type"] = kind
                flat.append(pattern)
    return flat


def pattern_text(pattern):
    parts = [
        "Pattern: {}".format(pattern.get("pattern_id", "")),
        "Type: {}".format(pattern.get("pattern_type", "")),
        "Description: {}".format(pattern.get("description", "")),
    ]
    if pattern.get("generation_strategy"):
        parts.append("Strategy: {}".format(pattern["generation_strategy"]))
    if pattern.get("example_input"):
        parts.append("Example Input: {}".format(pattern["example_input"]))
    if pattern.get("error_types_caught"):
        parts.append("Catches: {}".format(", ".join(pattern["error_types_caught"])))
    if pattern.get("pattern_type") == "boundary":
        parts.append("Boundary Type: {}".format(pattern.get("boundary_type", "")))
    elif pattern.get("pattern_type") == "adversarial":
        parts.append("Attack Vector: {}".format(pattern.get("attack_vector", "")))
        parts.append("Why Effective: {}".format(pattern.get("why_effective", "")))
    return "\n".join(parts)


def constraint_text(item):
    parts = ["Problem: {}".format(item.get("problem_id", ""))]
    if item.get("input_format"):
        parts.append("Input: {}".format(json.dumps(item["input_format"], ensure_ascii=False)))
    for value in (item.get("constraints") or {}).get("explicit", []):
        parts.append("Constraint: {}".format(json.dumps(value, ensure_ascii=False)))
    for value in item.get("implicit_constraints", []):
        parts.append("Implicit: {}".format(value.get("description", "")))
    for value in item.get("boundary_conditions", []):
        parts.append("Boundary: {}".format(json.dumps(value, ensure_ascii=False)))
    return "\n".join(parts)


def fit_and_save(name, texts):
    print("[build] {}: {} documents".format(name, len(texts)))
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2),
                                 stop_words="english", sublinear_tf=True)
    matrix = vectorizer.fit_transform(texts)
    svd = TruncatedSVD(n_components=min(DIMENSIONS, matrix.shape[1] - 1), random_state=42)
    svd.fit(matrix)
    with (OUTPUT / "tfidf_{}.pkl".format(name)).open("wb") as handle:
        pickle.dump({"vectorizer": vectorizer, "svd": svd, "fitted": True,
                     "collection": name}, handle, protocol=4)


def main():
    problems = load("problems_clean.json")
    constraints = [item for item in load("constraints.json") if "error" not in item]
    patterns = flatten_patterns(load("patterns.json"))
    adversarial = []
    for item in patterns:
        if item["pattern_type"] == "adversarial":
            legacy_entry = dict(item)
            legacy_entry.pop("pattern_type", None)
            adversarial.append(legacy_entry)
    fit_and_save("cf_problems", [problem_text(item) for item in problems])
    fit_and_save("cf_patterns", [pattern_text(item) for item in patterns])
    fit_and_save("cf_constraints", [constraint_text(item) for item in constraints])
    fit_and_save("cf_error_patterns", [pattern_text(item) for item in adversarial])


if __name__ == "__main__":
    main()

"""Query the released knowledge base from the command line."""
import argparse
import json

from kb_core.vector_store import VectorStore


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the OJ test-pattern knowledge base")
    parser.add_argument("query", help="A problem description or testing-related query")
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to return")
    args = parser.parse_args()

    store = VectorStore()
    results = store.retrieve_knowledge(args.query, top_k=args.top_k)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

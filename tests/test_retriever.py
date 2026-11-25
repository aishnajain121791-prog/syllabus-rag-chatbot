import sys
from pathlib import Path

# Ensure project root is on sys.path when executed as `python tests/test_retriever.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.retriever import search


def main() -> None:
    query = "What topics are in Unit 3?"
    print(f"Query: {query}\n")

    results = search(query, k=3)
    for i, item in enumerate(results, start=1):
        print(f"--- Result {i} ---")
        print(f"Score: {item['score']:.4f} | Source: {item['source']} | Page: {item['page']}")
        print(item["text"])
        print()


if __name__ == "__main__":
    main()

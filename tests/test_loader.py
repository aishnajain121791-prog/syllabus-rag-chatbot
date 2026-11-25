import sys
from pathlib import Path

# Ensure project root is on sys.path when executed as `python tests/test_loader.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import load_pdfs


def main() -> None:
    chunks = load_pdfs()
    print(f"Total chunks loaded: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:5], start=1):
        print(f"--- Sample Chunk {i} ---")
        print(f"Source: {chunk['source']} | Page: {chunk['page']} | Length: {len(chunk['text'])}")
        print(chunk["text"])
        print()


if __name__ == "__main__":
    main()

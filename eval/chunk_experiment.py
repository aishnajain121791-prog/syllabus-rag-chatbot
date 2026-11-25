import json
import sys
import tempfile
from pathlib import Path

# Allow running directly as `python eval/chunk_experiment.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.loader import load_pdfs
from src.retriever import build_index, search

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
RESULTS_PATH = Path(__file__).resolve().parent / "chunk_results.md"

CHUNK_SIZES = [500, 800, 1200]
OVERLAPS = [100, 150]


def _matches_expected(chunk: dict, expected_source: str | None, expected_page: int) -> bool:
    """Return True if a retrieved chunk came from the expected source and page."""
    if chunk.get("page") != expected_page:
        return False
    if expected_source and chunk.get("source") != expected_source:
        return False
    return True


def evaluate_config(
    questions: list[dict],
    chunk_size: int,
    overlap: int,
    temp_root: Path,
) -> dict:
    """Build a temporary FAISS index for (chunk_size, overlap) and compute retrieval metrics."""
    chunks = load_pdfs(chunk_size=chunk_size, overlap=overlap)
    exp_index_dir = temp_root / f"index_{chunk_size}_{overlap}"
    build_index(chunks=chunks, index_dir=exp_index_dir)

    total_questions = len(questions)
    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_ranks: list[float] = []
    matching_chunks = 0
    total_retrieved = 0

    for item in questions:
        retrieved = search(item["question"], k=3, index_dir=exp_index_dir)
        expected_source = item.get("source")
        expected_page = item["page"]

        matches = [
            _matches_expected(chunk, expected_source, expected_page)
            for chunk in retrieved
        ]
        if matches and matches[0]:
            hit_at_1 += 1
        if any(matches):
            hit_at_3 += 1

        rr = 0.0
        for rank, is_match in enumerate(matches, start=1):
            if is_match:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        matching_chunks += sum(1 for m in matches if m)
        total_retrieved += len(retrieved)

    return {
        "chunk_size": chunk_size,
        "overlap": overlap,
        "num_chunks": len(chunks),
        "hit1": hit_at_1 / total_questions if total_questions else 0.0,
        "hit3": hit_at_3 / total_questions if total_questions else 0.0,
        "mrr": sum(reciprocal_ranks) / total_questions if total_questions else 0.0,
        "chunk_acc": matching_chunks / total_retrieved if total_retrieved else 0.0,
        "hit1_raw": f"{hit_at_1}/{total_questions}",
        "hit3_raw": f"{hit_at_3}/{total_questions}",
        "chunk_acc_raw": f"{matching_chunks}/{total_retrieved}",
    }


def format_markdown_table(rows: list[dict]) -> str:
    """Format experiment results as a GitHub-flavored Markdown table."""
    lines = [
        "# Chunk Size & Overlap Experiment Results",
        "",
        f"Evaluated on `{QUESTIONS_PATH.name}` ({len(json.loads(QUESTIONS_PATH.read_text(encoding='utf-8')))} questions, top-3 FAISS `IndexFlatIP` retrieval).",
        "",
        "| Chunk Size | Overlap | Total Chunks | Hit@1 | Hit@3 | MRR | Expected-Chunk Accuracy |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['chunk_size']} | {r['overlap']} | {r['num_chunks']} | "
            f"{r['hit1']*100:.1f}% ({r['hit1_raw']}) | "
            f"{r['hit3']*100:.1f}% ({r['hit3_raw']}) | "
            f"{r['mrr']:.4f} | "
            f"{r['chunk_acc']*100:.1f}% ({r['chunk_acc_raw']}) |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    all_questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = [q for q in all_questions if q.get("answerable", True)]
    rows = []

    with tempfile.TemporaryDirectory() as tmp_dir:
        temp_root = Path(tmp_dir)
        for chunk_size in CHUNK_SIZES:
            for overlap in OVERLAPS:
                rows.append(
                    evaluate_config(
                        questions=questions,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        temp_root=temp_root,
                    )
                )

    markdown = format_markdown_table(rows)
    RESULTS_PATH.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()

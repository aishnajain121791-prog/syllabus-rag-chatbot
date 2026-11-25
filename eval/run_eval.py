import json
import sys
from pathlib import Path

# Allow running directly as `python eval/run_eval.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.rag import MIN_SCORE, answer

QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
RESULTS_PATH = Path(__file__).resolve().parent / "results.md"

REFUSAL_PHRASES = (
    "i couldn't find this in your notes",
    "i don't know",
    "i do not know",
    "not in the context",
    "not in the provided context",
    "does not contain",
    "doesn't contain",
    "not mentioned in the",
    "no relevant context",
    "cannot answer",
)


def _matches_expected(chunk: dict, expected_source: str | None, expected_page: int) -> bool:
    """Return True if a retrieved chunk matches the expected page (and source if provided)."""
    if chunk.get("page") != expected_page:
        return False
    if expected_source and chunk.get("source") != expected_source:
        return False
    return True


def _classify_trap_response(result: dict) -> str:
    """Classify a trap-question result as 'threshold_refusal', 'llm_refusal', 'api_error', or 'answered'."""
    ans_lower = result.get("answer", "").strip().lower()
    err = result.get("error")

    if err in ("api_error", "missing_api_key", "model_not_found", "empty_response"):
        return "api_error"
    if ans_lower == "i couldn't find this in your notes" or err == "no_context":
        return "threshold_refusal"
    if any(phrase in ans_lower for phrase in REFUSAL_PHRASES):
        return "llm_refusal"
    return "answered"


def main() -> None:
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    answerable = [q for q in questions if q.get("answerable", True)]
    traps = [q for q in questions if not q.get("answerable", True)]

    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_ranks: list[float] = []
    total_matching_chunks = 0
    total_retrieved_chunks = 0
    answerable_api_errors = 0

    detail_lines: list[str] = []

    print(f"Running evaluation on {len(answerable)} answerable + {len(traps)} trap questions (MIN_SCORE={MIN_SCORE:.2f})...\n")

    for i, item in enumerate(answerable, start=1):
        question = item["question"]
        expected_source = item.get("source")
        expected_page = item["page"]

        result = answer(question)
        sources = result.get("sources", [])
        if result.get("error") in ("api_error", "missing_api_key", "model_not_found"):
            answerable_api_errors += 1

        matches = [
            _matches_expected(s, expected_source, expected_page) for s in sources
        ]
        is_hit1 = bool(matches and matches[0])
        is_hit3 = any(matches)
        if is_hit1:
            hit_at_1 += 1
        if is_hit3:
            hit_at_3 += 1

        rr = 0.0
        for rank, is_match in enumerate(matches, start=1):
            if is_match:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

        matching = sum(1 for m in matches if m)
        total_matching_chunks += matching
        total_retrieved_chunks += len(sources)

        expected_label = (
            f"{expected_source} (p.{expected_page})"
            if expected_source
            else f"p.{expected_page}"
        )
        retrieved_labels = ", ".join(
            f"{s['source']} (p.{s['page']}, {s['score']:.3f})" for s in sources
        )

        print(f"[Q{i}/{len(answerable)}] {question}")
        print(f"  Expected : {expected_label}")
        print(f"  Retrieved: {retrieved_labels or 'None'}")
        print(f"  Hit@1: {'YES' if is_hit1 else 'NO'} | Hit@3: {'YES' if is_hit3 else 'NO'} | RR: {rr:.2f} | Chunks: {matching}/{len(sources)}")
        print()

        detail_lines.append(
            f"| Q{i} | {expected_label} | {'YES' if is_hit1 else 'NO'} | "
            f"{'YES' if is_hit3 else 'NO'} | {rr:.2f} | {matching}/{len(sources) or 3} |"
        )

    # Evaluate trap (unanswerable) questions
    refused_threshold = 0
    refused_llm = 0
    trap_api_errors = 0
    trap_answered = 0
    trap_rows: list[str] = []

    for j, item in enumerate(traps, start=1):
        question = item["question"]
        result = answer(question)
        status = _classify_trap_response(result)
        sources = result.get("sources", [])
        top_score = f"{sources[0]['score']:.3f}" if sources else "<0.30"

        if status == "threshold_refusal":
            refused_threshold += 1
        elif status == "llm_refusal":
            refused_llm += 1
        elif status == "api_error":
            trap_api_errors += 1
        else:
            trap_answered += 1

        print(f"[Trap {j}/{len(traps)}] {question}")
        print(f"  Status   : {status} | Top Score: {top_score}")
        print(f"  Answer   : {result['answer'][:140].replace(chr(10), ' ')}")
        print()

        trap_rows.append(
            f"| T{j} | {question} | {status} | {top_score} |"
        )

    n_ans = len(answerable)
    n_traps = len(traps)
    hit1_pct = (hit_at_1 / n_ans * 100) if n_ans else 0.0
    hit3_pct = (hit_at_3 / n_ans * 100) if n_ans else 0.0
    mrr = (sum(reciprocal_ranks) / n_ans) if n_ans else 0.0
    chunk_acc = (total_matching_chunks / total_retrieved_chunks * 100) if total_retrieved_chunks else 0.0

    total_refused = refused_threshold + refused_llm
    refusal_rate = (total_refused / n_traps * 100) if n_traps else 0.0

    summary_lines = [
        "# Evaluation Results",
        "",
        f"- **Answerable Questions**: {n_ans}",
        f"- **Trap (Unanswerable) Questions**: {n_traps}",
        f"- **Similarity Threshold (`MIN_SCORE`)**: `{MIN_SCORE:.2f}`",
        "",
        "## Summary Metrics",
        "",
        "| Metric | Score |",
        "| --- | --- |",
        f"| Hit@1 | {hit1_pct:.1f}% ({hit_at_1}/{n_ans}) |",
        f"| Hit@3 | {hit3_pct:.1f}% ({hit_at_3}/{n_ans}) |",
        f"| MRR | {mrr:.4f} |",
        f"| Expected-Chunk Accuracy | {chunk_acc:.1f}% ({total_matching_chunks}/{total_retrieved_chunks}) |",
        f"| Trap Refusal Rate | {refusal_rate:.1f}% ({total_refused}/{n_traps}: {refused_threshold} threshold, {refused_llm} LLM) |",
        f"| Trap Non-Refusals / API Errors | {trap_answered} answered / {trap_api_errors} API errors (answerable API errors: {answerable_api_errors}) |",
        "",
        "## Answerable Questions Breakdown",
        "",
        "| ID | Expected Source & Page | Hit@1 | Hit@3 | RR | Matching Chunks |",
        "| --- | --- | :---: | :---: | ---: | ---: |",
        *detail_lines,
        "",
        "## Trap Questions Breakdown",
        "",
        "| ID | Question | Classification | Top Score Returned |",
        "| --- | --- | --- | ---: |",
        *trap_rows,
        "",
    ]
    report = "\n".join(summary_lines)
    RESULTS_PATH.write_text(report, encoding="utf-8")

    print("=== Evaluation Summary ===")
    print(f"Hit@1                   : {hit_at_1}/{n_ans} ({hit1_pct:.1f}%)")
    print(f"Hit@3                   : {hit_at_3}/{n_ans} ({hit3_pct:.1f}%)")
    print(f"MRR                     : {mrr:.4f}")
    print(f"Expected-Chunk Accuracy : {total_matching_chunks}/{total_retrieved_chunks} ({chunk_acc:.1f}%)")
    print(
        f"Trap Refusal Rate       : {total_refused}/{n_traps} ({refusal_rate:.1f}%) "
        f"[threshold={refused_threshold}, llm={refused_llm}, answered={trap_answered}, api_error={trap_api_errors}]"
    )
    print(f"\nSaved markdown report to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

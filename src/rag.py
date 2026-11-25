import logging
import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

try:
    from src.database import list_notes
    from src.retriever import get_model, load_or_build_index, search
except ModuleNotFoundError:
    from database import list_notes
    from retriever import get_model, load_or_build_index, search

ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
MIN_SCORE = 0.30
SYSTEM_PROMPT = (
    "Answer using only the provided context. If the answer is not in the "
    "context, say you don't know. Cite the source file and page number."
)
REWRITE_PROMPT = (
    "Given the recent conversation history and the user's latest follow-up message, "
    "rewrite the follow-up into a concise, self-contained standalone question for "
    "searching course notes. If the latest message is already standalone, return it "
    "unchanged. Return only the rewritten question."
)
DEFAULT_MODEL = "gemini-3.6-flash"

logger = logging.getLogger(__name__)
if not logger.handlers and not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")


def _clean_key(value: str | None) -> str:
    """Return a trimmed API key if non-empty and not the placeholder value, else ''."""
    cleaned = str(value or "").strip()
    if not cleaned or cleaned == "your_key_here":
        return ""
    return cleaned


def resolve_api_key(secrets_key: str = "", sidebar_key: str = "") -> str:
    """Resolve GEMINI_API_KEY: environment variables (.env/OS) -> st.secrets -> sidebar."""
    load_dotenv(dotenv_path=ENV_PATH)
    for candidate in (os.getenv("GEMINI_API_KEY"), secrets_key, sidebar_key):
        valid = _clean_key(candidate)
        if valid:
            return valid
    return ""


def is_api_key_configured(api_key: str | None = None) -> bool:
    """Return True if a non-placeholder GEMINI_API_KEY is available."""
    resolved = resolve_api_key() if api_key is None else _clean_key(api_key)
    return bool(resolved)


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks with source filename and page number for Gemini."""
    sections = []
    for i, chunk in enumerate(chunks, start=1):
        sections.append(
            f"[{i}] Source: {chunk['source']} | Page: {chunk['page']}\n{chunk['text']}"
        )
    return "\n\n".join(sections)


def _format_history(chat_history: list[dict]) -> str:
    """Format up to the last 4 chat messages into plain text."""
    lines = []
    for msg in chat_history[-4:]:
        role = "User" if msg.get("role") == "user" else "Assistant"
        content = str(msg.get("content", "")).strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def _is_model_not_found(exc: Exception) -> bool:
    """Return True if the exception indicates the Gemini model name was not found."""
    if isinstance(exc, genai_errors.APIError):
        if exc.code == 404 or str(exc.status or "").upper() == "NOT_FOUND":
            return True
    text = str(exc).lower()
    return "not found" in text or "no longer available" in text or "404" in text


def _rewrite_question(
    question: str,
    chat_history: list[dict],
    client: genai.Client | None,
    model_name: str,
) -> str:
    """Rewrite a follow-up question into a standalone query using recent chat history."""
    history_text = _format_history(chat_history)
    if not history_text or client is None:
        return question

    prompt = f"Conversation History:\n{history_text}\n\nFollow-up Question: {question}"
    try:
        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=REWRITE_PROMPT,
            ),
        )
        rewritten = (response.text or "").strip().strip('"').strip("'")
        return rewritten or question
    except Exception:
        return question


def _resolve_allowed_sources(
    note_filter: str | list[str] | tuple[str, ...] | set[str] | None,
) -> set[str] | None:
    """Resolve a note/subject filter into a set of allowed PDF filenames, or None."""
    if note_filter is None:
        return None

    if isinstance(note_filter, (list, tuple, set)):
        cleaned = {str(item).strip() for item in note_filter if str(item).strip()}
        return cleaned or None

    filter_str = str(note_filter).strip()
    if not filter_str or filter_str.lower() == "all notes":
        return None

    if filter_str.startswith("File: "):
        return {filter_str[len("File: ") :].strip()}

    notes = list_notes()
    if filter_str.startswith("Subject: "):
        target_subject = filter_str[len("Subject: ") :].strip().lower()
        return {
            note["filename"]
            for note in notes
            if note.get("subject", "").strip().lower() == target_subject
        }

    subject_matches = {
        note["filename"]
        for note in notes
        if note.get("subject", "").strip().lower() == filter_str.lower()
    }
    if subject_matches:
        return subject_matches

    return {filter_str}


def _retrieve_chunks(
    query: str,
    k: int = 3,
    note_filter: str | list[str] | tuple[str, ...] | set[str] | None = None,
) -> tuple[list[dict], bool]:
    """Retrieve top-k chunks and report whether any chunks matched the requested filter."""
    allowed_sources = _resolve_allowed_sources(note_filter)
    if allowed_sources is None:
        results = search(query, k=k)
        return results, bool(results)

    if k <= 0 or not allowed_sources:
        return [], False

    index, chunks = load_or_build_index()
    allowed_ids = [
        idx for idx, chunk in enumerate(chunks) if chunk.get("source") in allowed_sources
    ]
    if not allowed_ids:
        return [], False

    model = get_model()
    query_vec = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    query_vec = np.asarray(query_vec, dtype="float32")

    top_k = min(k, len(allowed_ids))
    selector = faiss.IDSelectorBatch(allowed_ids)
    params = faiss.SearchParameters(sel=selector)
    scores, indices = index.search(query_vec, top_k, params=params)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        results.append(
            {
                "text": chunk["text"],
                "score": float(score),
                "source": chunk["source"],
                "page": chunk["page"],
            }
        )
    return results, True


def answer(
    question: str,
    k: int = 3,
    chat_history: list[dict] | None = None,
    note_filter: str | list[str] | tuple[str, ...] | set[str] | None = None,
    api_key: str | None = None,
) -> dict:
    """Rewrite follow-up if needed, retrieve filtered top-k chunks, and answer with Gemini."""
    question = question.strip()
    if not question:
        return {
            "answer": "Please enter a non-empty question.",
            "sources": [],
            "error": "empty_question",
        }

    resolved_key = resolve_api_key() if api_key is None else _clean_key(api_key)
    model_name = os.getenv("GEMINI_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL
    client = (
        genai.Client(
            api_key=resolved_key,
            http_options=types.HttpOptions(timeout=20000),
        )
        if resolved_key
        else None
    )

    recent_history = (chat_history or [])[-4:]
    standalone_question = _rewrite_question(
        question=question,
        chat_history=recent_history,
        client=client,
        model_name=model_name,
    )

    try:
        chunks, has_filter_matches = _retrieve_chunks(
            standalone_question,
            k=k,
            note_filter=note_filter,
        )
    except Exception as exc:
        return {
            "answer": f"No indexed syllabus chunks are available yet ({exc}). Please upload readable PDFs in My Notes.",
            "sources": [],
            "rewritten_question": standalone_question,
            "error": "retrieval_error",
        }

    if not chunks:
        if note_filter and str(note_filter).strip().lower() != "all notes" and not has_filter_matches:
            msg = (
                "No indexed content matched the selected note/subject filter "
                "(it may have been deleted or contain no extractable text)."
            )
        else:
            msg = "I don't know. No relevant context was found in the syllabus documents."
        return {
            "answer": msg,
            "sources": [],
            "rewritten_question": standalone_question,
            "error": "no_context",
        }

    best_score = max(chunk["score"] for chunk in chunks)
    logger.info(
        "Top retrieval score: %.4f (MIN_SCORE=%.2f) for query=%r",
        best_score,
        MIN_SCORE,
        standalone_question,
    )
    if best_score < MIN_SCORE:
        logger.info(
            "Best score %.4f < MIN_SCORE %.2f; skipping LLM call.",
            best_score,
            MIN_SCORE,
        )
        return {
            "answer": "I couldn't find this in your notes",
            "sources": [],
            "rewritten_question": standalone_question,
        }

    sources = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "score": chunk["score"],
            "text": chunk["text"],
        }
        for chunk in chunks
    ]

    if client is None:
        return {
            "answer": (
                "GEMINI_API_KEY is missing or not configured. "
                "Copy .env.example to .env and add your key to enable Gemini answers."
            ),
            "sources": sources,
            "rewritten_question": standalone_question,
            "error": "missing_api_key",
        }

    context = _format_context(chunks)
    if standalone_question != question:
        user_prompt = (
            f"Context:\n{context}\n\n"
            f"Standalone Question: {standalone_question}\n"
            f"User Request: {question}"
        )
    else:
        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"

    try:
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
            ),
        )
        answer_text = (response.text or "").strip()
        if not answer_text:
            return {
                "answer": "I don't know. Gemini returned an empty response.",
                "sources": sources,
                "rewritten_question": standalone_question,
                "error": "empty_response",
            }
    except Exception as exc:
        if _is_model_not_found(exc):
            return {
                "answer": (
                    f"Gemini model '{model_name}' was not found or is unavailable. "
                    "Check GEMINI_MODEL in .env (for example, 'gemini-3.8-flash')."
                ),
                "sources": sources,
                "rewritten_question": standalone_question,
                "error": "model_not_found",
            }
        return {
            "answer": f"Gemini API error: {exc}",
            "sources": sources,
            "rewritten_question": standalone_question,
            "error": "api_error",
        }

    return {
        "answer": answer_text,
        "sources": sources,
        "rewritten_question": standalone_question,
    }

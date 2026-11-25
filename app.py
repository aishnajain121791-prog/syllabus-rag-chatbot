import shutil
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import (
    add_message,
    create_session,
    delete_note,
    delete_session,
    get_messages,
    init_db,
    list_notes,
    list_sessions,
    upsert_note,
)
from src.loader import inspect_pdf
from src.rag import answer, is_api_key_configured, resolve_api_key
from src.retriever import DEFAULT_INDEX_DIR, build_index, clear_index, get_model

DATA_DIR = PROJECT_ROOT / "data"
SAMPLE_DATA_DIR = PROJECT_ROOT / "sample_data"
DEMO_SUBJECT = "Sample Notes"
DEMO_QUESTION_LIMIT = 20
MAX_FILE_SIZE_MB = 10
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_TOTAL_FILES = 5
QUIZ_ANSWER_MARKER = "---ANSWERS---"

st.set_page_config(
    page_title="Syllabus Q&A Chatbot",
    page_icon="📚",
    layout="centered",
)


def _ensure_app_storage() -> None:
    """Create data/, faiss_index/, and SQLite tables on first run if missing."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    DEFAULT_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    init_db()


def _get_secret_api_key() -> str:
    """Read GEMINI_API_KEY from st.secrets safely without failing if secrets.toml is absent."""
    try:
        return str(st.secrets.get("GEMINI_API_KEY", "")).strip()
    except Exception:
        return ""


@st.cache_resource
def get_cached_embedding_model():
    """Cache the SentenceTransformer model across Streamlit reruns."""
    return get_model()


def _seed_sample_notes() -> None:
    """Copy sample_data/ PDFs into data/, register them in SQLite, and build FAISS index."""
    if not SAMPLE_DATA_DIR.exists():
        return

    sample_pdfs = sorted(
        p for p in SAMPLE_DATA_DIR.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"
    )
    if not sample_pdfs:
        return

    seeded_meta: list[tuple[str, int]] = []
    for sample_pdf in sample_pdfs:
        info = inspect_pdf(sample_pdf)
        if info["status"] != "ok":
            continue
        dest_path = DATA_DIR / sample_pdf.name
        shutil.copy2(sample_pdf, dest_path)
        seeded_meta.append((sample_pdf.name, info["num_pages"]))

    if not seeded_meta:
        return

    _, all_chunks = build_index()
    for fname, num_pages in seeded_meta:
        file_chunks = sum(1 for chunk in all_chunks if chunk["source"] == fname)
        upsert_note(
            filename=fname,
            subject=DEMO_SUBJECT,
            num_pages=num_pages,
            num_chunks=file_chunks,
        )


_ensure_app_storage()

if "current_session_id" not in st.session_state:
    st.session_state["current_session_id"] = None
if "question_count" not in st.session_state:
    st.session_state["question_count"] = 0

st.info("Demo mode: uploaded files are temporary and may reset.")

# First-load initialization: warm cached embedding model and seed sample notes if DB is empty
_index_missing = (
    not (DEFAULT_INDEX_DIR / "index.faiss").exists()
    or not (DEFAULT_INDEX_DIR / "chunks.json").exists()
)
_needs_seed = (
    not list_notes()
    and not st.session_state.get("demo_cleared", False)
    and SAMPLE_DATA_DIR.exists()
    and any(p.is_file() and p.suffix.lower() == ".pdf" for p in SAMPLE_DATA_DIR.iterdir())
)
if not st.session_state.get("model_warmed", False) or _needs_seed or _index_missing:
    with st.spinner("Loading embedding model and building search index..."):
        get_cached_embedding_model()
        st.session_state["model_warmed"] = True
        if _needs_seed:
            _seed_sample_notes()
        elif _index_missing and any(
            p.is_file() and p.suffix.lower() == ".pdf" for p in DATA_DIR.iterdir()
        ):
            build_index()

# --- Sidebar: Navigation, API key override, and Chat session management ---
with st.sidebar:
    st.title("📚 Syllabus RAG")
    page = st.radio("Navigate", ["Chat", "My Notes"])

    sidebar_api_key = st.text_input(
        "Gemini API Key (optional)",
        type="password",
        placeholder="Uses env or st.secrets if blank",
        help="Fallback key for this session if GEMINI_API_KEY is not set in env/.env or st.secrets. Entering a key also removes the 20-question session limit.",
    )
    using_own_key = bool(sidebar_api_key.strip())
    limit_reached = (not using_own_key) and (
        st.session_state["question_count"] >= DEMO_QUESTION_LIMIT
    )
    if using_own_key:
        st.caption("Custom API key active (no session limit).")
    else:
        st.caption(
            f"Demo questions used: {st.session_state['question_count']}/{DEMO_QUESTION_LIMIT}"
        )

    resolved_api_key = resolve_api_key(
        secrets_key=_get_secret_api_key(),
        sidebar_key=sidebar_api_key,
    )

    if page == "Chat":
        st.divider()
        if st.button("➕ New chat", use_container_width=True):
            st.session_state["current_session_id"] = None
            st.rerun()

        st.subheader("Previous Chats")
        sessions = list_sessions()
        if not sessions:
            st.caption("No saved chats yet.")
        else:
            for sess in sessions:
                col_open, col_del = st.columns([4, 1])
                is_active = sess["id"] == st.session_state["current_session_id"]
                if col_open.button(
                    sess["title"],
                    key=f"open_session_{sess['id']}",
                    type="primary" if is_active else "secondary",
                    use_container_width=True,
                ):
                    st.session_state["current_session_id"] = sess["id"]
                    st.rerun()

                if col_del.button(
                    "🗑️",
                    key=f"delete_session_{sess['id']}",
                    help="Delete chat",
                ):
                    delete_session(sess["id"])
                    if st.session_state["current_session_id"] == sess["id"]:
                        st.session_state["current_session_id"] = None
                    st.rerun()


def _existing_pdf_names() -> set[str]:
    """Return the set of PDF filenames currently stored in data/."""
    if not DATA_DIR.exists():
        return set()
    return {
        p.name
        for p in DATA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() == ".pdf"
    }


def _rebuild_or_clear_index() -> int:
    """Rebuild the FAISS index from data/, or clear saved and cached index if data/ is empty."""
    if _existing_pdf_names():
        try:
            _, chunks = build_index()
            return len(chunks)
        except ValueError:
            clear_index()
            return 0

    clear_index()
    return 0


def _render_sources_expander(sources: list[dict]) -> None:
    """Render retrieved source citations inside an expandable panel."""
    with st.expander("Sources"):
        if sources:
            for i, src in enumerate(sources, start=1):
                score = src.get("score")
                score_text = (
                    f" (Score: {score:.4f})" if isinstance(score, (int, float)) else ""
                )
                st.markdown(
                    f"**{i}. {src['source']}** — Page {src['page']}{score_text}"
                )
                st.write(src["text"])
                if i < len(sources):
                    st.divider()
        else:
            st.write("No sources were retrieved.")


def _make_session_title(first_question: str, max_len: int = 45) -> str:
    """Build a concise session title from the first user question."""
    clean = " ".join(first_question.split())
    if len(clean) <= max_len:
        return clean
    return clean[: max_len - 3] + "..."


def _get_filter_options() -> list[str]:
    """Build dropdown options for filtering retrieval by user-uploaded subject or PDF file."""
    options = ["All notes"]
    notes = list_notes()
    disk_files = _existing_pdf_names()
    uploaded_notes = [note for note in notes if note.get("filename") in disk_files]

    subjects = sorted(
        {
            note["subject"].strip()
            for note in uploaded_notes
            if note.get("subject") and note["subject"].strip()
        }
    )
    for subj in subjects:
        options.append(f"Subject: {subj}")

    uploaded_files = sorted(
        {note["filename"] for note in uploaded_notes if note.get("filename")}
    )
    for fname in uploaded_files:
        options.append(f"File: {fname}")

    return options


def _render_assistant_content(content: str) -> None:
    """Render assistant output, hiding quiz answers in an expander when present."""
    if QUIZ_ANSWER_MARKER in content:
        questions_part, answers_part = content.split(QUIZ_ANSWER_MARKER, 1)
        st.write(questions_part.strip())
        with st.expander("Show Quiz Answers"):
            st.write(answers_part.strip())
    else:
        st.write(content)


def _format_chat_export(messages: list[dict]) -> str:
    """Format chat messages and citations as plain text for .txt download."""
    lines = ["Syllabus Q&A Chatbot - Chat Export", "=" * 36, ""]
    for msg in messages:
        role = "USER" if msg["role"] == "user" else "ASSISTANT"
        content = msg["content"].replace(QUIZ_ANSWER_MARKER, "\nAnswers:\n").strip()
        lines.append(f"[{role}] ({msg.get('created_at', '')})")
        lines.append(content)
        sources = msg.get("sources") or []
        if sources:
            src_refs = ", ".join(f"{s['source']} (p.{s['page']})" for s in sources)
            lines.append(f"Sources: {src_refs}")
        lines.append("")
    return "\n".join(lines)


def _find_deleted_referenced_files(messages: list[dict]) -> list[str]:
    """Return sorted filenames cited in `messages` that no longer exist in data/."""
    existing = _existing_pdf_names()
    missing = {
        src["source"]
        for msg in messages
        for src in (msg.get("sources") or [])
        if src.get("source") and src["source"] not in existing
    }
    return sorted(missing)


def _run_and_save_turn(
    session_id: int | None,
    user_display: str,
    rag_prompt: str,
    note_filter: str,
    chat_history: list[dict] | None = None,
    k: int = 3,
) -> None:
    """Run RAG for a user prompt or study action, save valid responses, and surface errors cleanly."""
    with st.spinner("Searching syllabus and generating response..."):
        get_cached_embedding_model()
        result = answer(
            rag_prompt,
            k=k,
            chat_history=chat_history,
            note_filter=note_filter,
            api_key=resolved_api_key,
        )

    error_kind = result.get("error")
    if error_kind:
        if error_kind in ("missing_api_key", "no_context"):
            st.warning(result["answer"])
        else:
            st.error(result["answer"])
        if result.get("sources"):
            _render_sources_expander(result["sources"])
        return

    if session_id is None:
        session_id = create_session(title=_make_session_title(user_display))
        st.session_state["current_session_id"] = session_id

    add_message(session_id=session_id, role="user", content=user_display)
    add_message(
        session_id=session_id,
        role="assistant",
        content=result["answer"],
        sources=result.get("sources", []),
    )
    st.session_state["question_count"] += 1
    st.rerun()


def render_chat_page() -> None:
    """Render the multi-turn chat interface backed by SQLite sessions."""
    st.title("📚 Syllabus Q&A Chatbot")
    st.write("Ask questions about your uploaded course syllabus and notes.")

    if not is_api_key_configured(api_key=resolved_api_key):
        st.warning(
            "GEMINI_API_KEY is missing or not configured. "
            "Set it in environment variables / .env, st.secrets, or the sidebar input."
        )

    if not _existing_pdf_names():
        st.info("No PDFs found in `data/`. Upload notes in **My Notes** to start asking questions.")

    session_id = st.session_state["current_session_id"]
    messages = get_messages(session_id) if session_id is not None else []

    missing_files = _find_deleted_referenced_files(messages)
    if missing_files:
        st.warning(
            "Some notes referenced in this chat have been deleted: "
            + ", ".join(missing_files)
            + ". New follow-up questions will only search remaining notes."
        )

    if not messages:
        st.info("Ask a question below to start a new chat.")

    for msg in messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant":
                _render_assistant_content(msg["content"])
                _render_sources_expander(msg.get("sources", []))
            else:
                st.write(msg["content"])

    filter_options = _get_filter_options()
    if (
        "chat_note_filter" in st.session_state
        and st.session_state["chat_note_filter"] not in filter_options
    ):
        st.session_state["chat_note_filter"] = "All notes"

    note_filter = st.selectbox(
        "Search scope",
        options=filter_options,
        index=0,
        key="chat_note_filter",
    )

    # --- Study action buttons for the selected note + Download chat ---
    note_selected = note_filter != "All notes"
    study_disabled = (not note_selected) or limit_reached
    btn_col1, btn_col2, btn_col3, btn_col4 = st.columns(4)

    summarize_clicked = btn_col1.button(
        "Summarize this note",
        disabled=study_disabled,
        help="Select a specific subject or file above first",
        use_container_width=True,
    )
    quiz_clicked = btn_col2.button(
        "Generate 5 quiz questions",
        disabled=study_disabled,
        help="Select a specific subject or file above first",
        use_container_width=True,
    )
    terms_clicked = btn_col3.button(
        "Key terms",
        disabled=study_disabled,
        help="Select a specific subject or file above first",
        use_container_width=True,
    )
    btn_col4.download_button(
        "Download chat as .txt",
        data=_format_chat_export(messages),
        file_name=f"syllabus_chat_{session_id or 'empty'}.txt",
        mime="text/plain",
        disabled=not messages,
        use_container_width=True,
    )

    if limit_reached:
        st.warning(
            f"You have reached the demo limit of {DEMO_QUESTION_LIMIT} questions for this "
            "browser session. Paste your own Gemini API key in the sidebar to remove this limit."
        )

    if summarize_clicked:
        _run_and_save_turn(
            session_id=session_id,
            user_display=f"Summarize this note ({note_filter})",
            rag_prompt=(
                "Provide a concise bullet-point summary of the main topics covered "
                "in the provided context. Cite the source file and page numbers."
            ),
            note_filter=note_filter,
            k=8,
        )

    if quiz_clicked:
        _run_and_save_turn(
            session_id=session_id,
            user_display=f"Generate 5 quiz questions ({note_filter})",
            rag_prompt=(
                "Generate 5 quiz questions based strictly on the provided context. "
                f"Format your response into two sections separated by a line containing only {QUIZ_ANSWER_MARKER}. "
                "In the first section, list only the 5 numbered questions without answers. "
                f"After {QUIZ_ANSWER_MARKER}, list the 5 numbered answers with brief explanations "
                "and cite the source file and page number for each."
            ),
            note_filter=note_filter,
            k=8,
        )

    if terms_clicked:
        _run_and_save_turn(
            session_id=session_id,
            user_display=f"Key terms ({note_filter})",
            rag_prompt=(
                "Extract a bulleted list of the important technical terms from the "
                "provided context, each with a clear one-line definition and source "
                "file and page citation."
            ),
            note_filter=note_filter,
            k=8,
        )

    if prompt := st.chat_input(
        "Ask a question about your syllabus...",
        disabled=limit_reached,
    ):
        clean_prompt = prompt.strip()
        if not clean_prompt:
            st.warning("Please enter a non-empty question.")
            return

        _run_and_save_turn(
            session_id=session_id,
            user_display=clean_prompt,
            rag_prompt=clean_prompt,
            note_filter=note_filter,
            chat_history=messages[-4:],
            k=3,
        )


def render_my_notes_page() -> None:
    """Render the My Notes management page (upload, list, and delete PDFs)."""
    st.title("🗂️ My Notes")
    st.write("Upload course PDFs, tag them by subject, and manage your indexed notes.")

    subject = st.text_input(
        "Subject name",
        placeholder="e.g., Computer Organization",
    )
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )
    st.caption(
        f"Demo limits: max {MAX_FILE_SIZE_MB} MB per file, up to {MAX_TOTAL_FILES} files total."
    )

    if st.button("Upload & Index", type="primary", disabled=not uploaded_files):
        with st.spinner("Validating PDFs, saving notes, and rebuilding FAISS index..."):
            try:
                DATA_DIR.mkdir(parents=True, exist_ok=True)
                clean_subject = subject.strip() or "General"
                existing_note_names = {n["filename"] for n in list_notes()}
                planned_new_count = 0

                # Deduplicate within the same upload batch while preserving order
                unique_uploads: dict[str, object] = {}
                for uploaded_file in uploaded_files:
                    unique_uploads[Path(uploaded_file.name).name] = uploaded_file

                valid_files: list[tuple[str, int, int]] = []
                skipped_messages: list[str] = []
                replaced_names: list[str] = []

                for safe_name, uploaded_file in unique_uploads.items():
                    file_bytes = uploaded_file.getbuffer()
                    file_size = getattr(uploaded_file, "size", None) or len(file_bytes)
                    if file_size > MAX_FILE_SIZE_BYTES:
                        skipped_messages.append(
                            f"**{safe_name}**: exceeds the {MAX_FILE_SIZE_MB} MB per-file limit "
                            f"({file_size / (1024 * 1024):.1f} MB)."
                        )
                        continue

                    is_new_file = safe_name not in existing_note_names
                    if (
                        is_new_file
                        and len(existing_note_names) + planned_new_count >= MAX_TOTAL_FILES
                    ):
                        skipped_messages.append(
                            f"**{safe_name}**: exceeds the demo limit of {MAX_TOTAL_FILES} files total. "
                            "Delete an existing note first."
                        )
                        continue

                    dest_path = DATA_DIR / safe_name
                    had_previous_file = dest_path.exists()
                    previous_bytes = dest_path.read_bytes() if had_previous_file else None

                    dest_path.write_bytes(file_bytes)
                    info = inspect_pdf(dest_path)

                    if info["status"] != "ok":
                        # Restore previous valid file if overwriting, or remove invalid upload
                        if had_previous_file and previous_bytes is not None:
                            dest_path.write_bytes(previous_bytes)
                        elif dest_path.exists():
                            dest_path.unlink()
                        skipped_messages.append(f"**{safe_name}**: {info['error']}")
                        continue

                    if is_new_file:
                        planned_new_count += 1
                    valid_files.append((safe_name, info["num_pages"], info["num_chunks"]))

                if valid_files:
                    total_chunks = _rebuild_or_clear_index()
                    st.session_state["demo_cleared"] = False
                    for safe_name, num_pages, num_chunks in valid_files:
                        _, was_duplicate = upsert_note(
                            filename=safe_name,
                            subject=clean_subject,
                            num_pages=num_pages,
                            num_chunks=num_chunks,
                        )
                        if was_duplicate:
                            replaced_names.append(safe_name)

                    st.success(
                        f"Indexed {len(valid_files)} PDF(s) ({total_chunks} total chunks in index)."
                    )
                    if replaced_names:
                        st.info(
                            "Updated existing record(s) for duplicate upload(s): "
                            + ", ".join(replaced_names)
                        )

                for msg in skipped_messages:
                    st.warning(f"Skipped {msg}")
            except Exception as exc:
                st.error(f"Failed to upload and index notes: {exc}")

    st.divider()

    st.subheader("Uploaded Notes")
    notes = list_notes()

    if not notes:
        st.info("No notes recorded yet. Upload one or more PDFs above.")
        return

    header_cols = st.columns([3, 2, 2, 1, 1.2])
    header_cols[0].markdown("**Name**")
    header_cols[1].markdown("**Subject**")
    header_cols[2].markdown("**Date**")
    header_cols[3].markdown("**Pages**")
    header_cols[4].markdown("**Action**")

    for note in notes:
        row_cols = st.columns([3, 2, 2, 1, 1.2])
        row_cols[0].write(note["filename"])
        row_cols[1].write(note["subject"] or "—")
        row_cols[2].write(str(note.get("uploaded_at") or "")[:10])
        row_cols[3].write(note["num_pages"])

        if row_cols[4].button("Delete", key=f"delete_note_{note['id']}"):
            with st.spinner(f"Deleting {note['filename']} and rebuilding index..."):
                try:
                    file_path = DATA_DIR / Path(note["filename"]).name
                    if file_path.exists():
                        file_path.unlink()

                    delete_note(note["id"])
                    if not list_notes():
                        st.session_state["demo_cleared"] = True
                    remaining_chunks = _rebuild_or_clear_index()
                    if (
                        st.session_state.get("chat_note_filter")
                        == f"File: {note['filename']}"
                    ):
                        st.session_state["chat_note_filter"] = "All notes"

                    st.session_state["delete_notice"] = (
                        f"Deleted {note['filename']} ({remaining_chunks} chunks remaining in index)."
                    )
                    st.rerun()
                except Exception as exc:
                    st.error(f"Failed to delete note: {exc}")

    if "delete_notice" in st.session_state:
        st.success(st.session_state.pop("delete_notice"))


if page == "Chat":
    render_chat_page()
else:
    render_my_notes_page()

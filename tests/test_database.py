import sys
from pathlib import Path

# Ensure project root is on sys.path when executed as `python tests/test_database.py`
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.database import (
    add_message,
    add_note,
    create_session,
    delete_note,
    delete_session,
    get_messages,
    list_notes,
    list_sessions,
)


def main() -> None:
    # 1. Notes CRUD
    note_id = add_note(
        filename="COMPUTER ORGANIZATION NOTES.pdf",
        subject="Computer Organization",
        num_pages=78,
        num_chunks=351,
    )
    print(f"add_note -> id={note_id}")

    notes = list_notes()
    print(f"list_notes -> count={len(notes)}, latest={notes[0]}")
    assert any(n["id"] == note_id for n in notes)

    deleted_note = delete_note(note_id)
    print(f"delete_note({note_id}) -> {deleted_note}")
    assert deleted_note is True
    assert all(n["id"] != note_id for n in list_notes())

    # 2. Chat sessions & messages CRUD
    session_id = create_session(title="Unit 3 Revision")
    print(f"\ncreate_session -> id={session_id}")

    sessions = list_sessions()
    print(f"list_sessions -> count={len(sessions)}, latest={sessions[0]}")
    assert any(s["id"] == session_id for s in sessions)

    user_msg_id = add_message(
        session_id=session_id,
        role="user",
        content="What topics are in Unit 3?",
    )
    assistant_msg_id = add_message(
        session_id=session_id,
        role="assistant",
        content="Unit 3 covers CPU organization and Control unit design.",
        sources=[{"source": "COMPUTER ORGANIZATION NOTES.pdf", "page": 47}],
    )
    print(f"add_message -> user_msg_id={user_msg_id}, assistant_msg_id={assistant_msg_id}")

    msgs = get_messages(session_id)
    print(f"get_messages({session_id}) -> count={len(msgs)}")
    for msg in msgs:
        print(f"  [{msg['role']}] {msg['content']} | sources={msg['sources']}")
    assert len(msgs) == 2
    assert msgs[1]["sources"][0]["page"] == 47

    deleted_session = delete_session(session_id)
    print(f"\ndelete_session({session_id}) -> {deleted_session}")
    assert deleted_session is True
    assert all(s["id"] != session_id for s in list_sessions())
    assert get_messages(session_id) == []

    print("\nAll database checks passed!")


if __name__ == "__main__":
    main()

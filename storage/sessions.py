import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .models import Session, Message


BASE_DIR = Path(__file__).resolve().parent.parent
OUT_DIR = BASE_DIR / "out"
SESSIONS_DIR = OUT_DIR / "sessions"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any = None) -> Any:
    if default is None:
        default = {}
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)


def create_session() -> Session:
    ensure_dirs()
    session_id = uuid.uuid4().hex
    session = Session(
        id=session_id,
        created_at=now_iso(),
        messages=[]
    )
    save_session(session)
    return session


def load_session(session_id: str) -> Session:
    path = session_path(session_id)
    if path.exists():
        data = read_json(path, {})
        messages = [Message(**msg) for msg in data.get("messages", [])]
        return Session(
            id=session_id,
            created_at=data.get("created_at", now_iso()),
            messages=messages
        )
    return create_session()


def save_session(session: Session) -> None:
    ensure_dirs()
    data = {
        "id": session.id,
        "created_at": session.created_at,
        "messages": [msg.model_dump() for msg in session.messages]
    }
    write_json(session_path(session.id), data)


def get_next_turn(session: Session) -> int:
    return sum(1 for msg in session.messages if msg.role == "user") + 1


def add_message(session: Session, role: str, content: str, turn: int, ts: Optional[str] = None) -> None:
    timestamp = ts if ts is not None else now_iso()
    message = Message(role=role, content=content, ts=timestamp, turn=turn)
    session.messages.append(message)
    save_session(session)


def clear_all_sessions() -> None:
    ensure_dirs()
    for session_file in SESSIONS_DIR.glob("*.json"):
        session_file.unlink()

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from .models import Memory
from .sessions import read_json, write_json, BASE_DIR, OUT_DIR


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


MEMORIES_FILE = OUT_DIR / "memories.json"


def load_all_memories() -> List[Dict[str, Any]]:
    return read_json(MEMORIES_FILE, [])


def save_all_memories(memories: List[Dict[str, Any]]) -> None:
    write_json(MEMORIES_FILE, memories)


def append_memories(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    all_memories = load_all_memories()
    all_memories.extend(records)
    save_all_memories(all_memories)
    return records


def create_memory_record(
    session_id: str,
    turn: int,
    index: int,
    text: str,
    user_text: str,
    assistant_text: str
) -> Dict[str, Any]:
    return {
        "id": f"{session_id}:{turn}:{index}",
        "text": text,
        "ts": now_iso(),
        "session_id": session_id,
        "turn": turn,
        "provenance": {"user": user_text, "assistant": assistant_text},
    }


def get_memories_for_session(session_id: str) -> List[Memory]:
    all_memories = load_all_memories()
    session_memories = [
        Memory(**mem) for mem in all_memories
        if mem.get("session_id") == session_id
    ]
    return session_memories


def get_recent_memories(limit: int = 8, session_id: Optional[str] = None) -> List[Memory]:
    all_memories = load_all_memories()
    if session_id:
        all_memories = [mem for mem in all_memories if mem.get("session_id") == session_id]
    sliced = all_memories[-limit:]
    return [Memory(**mem) for mem in reversed(sliced)]


def get_memory_count(session_id: Optional[str] = None) -> int:
    all_memories = load_all_memories()
    if session_id:
        all_memories = [mem for mem in all_memories if mem.get("session_id") == session_id]
    return len(all_memories)

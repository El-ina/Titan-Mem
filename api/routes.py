from pathlib import Path
from typing import Optional
import shutil

from fastapi import APIRouter, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse, HTMLResponse

from storage.models import ChatRequest
from storage.sessions import create_session, load_session, ensure_dirs
from storage.memories import get_recent_memories, get_memory_count
from chat.conversation import send_chat_message
from extraction.extractor import extract_atomic_memories
from extraction.adapters import get_extraction_adapter
from storage.memories import (
    create_memory_record,
    load_all_memories,
    save_all_memories,
)
from knowledge_graph.builder import build_graph
from storage.sessions import OUT_DIR


GRAPH_FILE = OUT_DIR / "graph.html"
MEMORIES_FILE = OUT_DIR / "memories.json"
SESSIONS_DIR = OUT_DIR / "sessions"


def run_memory_pipeline(session_id: str, turn: int, user_text: str, assistant_text: str) -> None:
    adapter = get_extraction_adapter()
    extracted = extract_atomic_memories(user_text, assistant_text, adapter)

    if not extracted:
        return

    records = [
        create_memory_record(session_id, turn, idx, text, user_text, assistant_text)
        for idx, text in enumerate(extracted)
    ]

    all_memories = load_all_memories()
    all_memories.extend(records)
    save_all_memories(all_memories)

    build_graph()


router = APIRouter()


@router.get("/")
def index() -> HTMLResponse:
    from pathlib import Path
    html_path = Path(__file__).parent.parent / "ui" / "static" / "index.html"
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@router.get("/graph")
def graph() -> FileResponse:
    if not GRAPH_FILE.exists():
        raise HTTPException(status_code=404, detail="Graph not generated yet. Start chatting to create memories!")
    return FileResponse(GRAPH_FILE)


@router.get("/api/session")
def create_session_route() -> dict:
    session = create_session()
    return {"session_id": session.id}


@router.get("/api/history")
def get_history(session_id: str) -> dict:
    ensure_dirs()
    session = load_session(session_id)
    messages = [{"role": msg.role, "content": msg.content} for msg in session.messages]
    return {"messages": messages}


@router.get("/api/memories")
def get_memories(session_id: Optional[str] = None, limit: int = 8) -> dict:
    ensure_dirs()
    memories = get_recent_memories(limit=limit, session_id=session_id)
    count = get_memory_count(session_id=session_id)
    return {
        "memories": [{"text": mem.text} for mem in memories],
        "count": count
    }


@router.post("/api/chat")
def chat(req: ChatRequest, background_tasks: BackgroundTasks) -> dict:
    ensure_dirs()
    session, assistant_text = send_chat_message(req.session_id, req.message)

    background_tasks.add_task(
        run_memory_pipeline,
        session.id,
        len([m for m in session.messages if m.role == "user"]),
        req.message,
        assistant_text
    )

    return {
        "session_id": session.id,
        "assistant": assistant_text,
        "memory_status": "queued",
    }


@router.post("/api/clear-memories")
def clear_memories() -> dict:
    """Clear all memories, sessions, and graphs"""
    ensure_dirs()
    
    # Clear the memories file
    if MEMORIES_FILE.exists():
        MEMORIES_FILE.write_text("[]", encoding="utf-8")
    
    # Remove the graph file
    if GRAPH_FILE.exists():
        GRAPH_FILE.unlink()
    
    # Clear all session files
    if SESSIONS_DIR.exists():
        # Remove all .json files in sessions directory
        for session_file in SESSIONS_DIR.glob("*.json"):
            session_file.unlink()
    
    # Also clear artifacts if it exists
    artifacts_file = OUT_DIR / "artifacts.json"
    if artifacts_file.exists():
        artifacts_file.unlink()
    
    return {
        "status": "success", 
        "message": "All memories, sessions, and graphs cleared"
    }
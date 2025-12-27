import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import networkx as nx
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel
from pyvis.network import Network

from build_graph import build_similarity_edges, ollama_embed

BASE_DIR = Path(__file__).resolve().parent
OUT_DIR = BASE_DIR / "out"
SESSIONS_DIR = OUT_DIR / "sessions"
MEMORIES_FILE = OUT_DIR / "memories.json"

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.1:8b")
EXTRACT_MODEL = os.getenv("OLLAMA_EXTRACT_MODEL", CHAT_MODEL)
EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text:v1.5")
CHAT_BACKEND = os.getenv("CHAT_BACKEND", "ollama")
SIM_TOP_K = int(os.getenv("SIM_TOP_K", "2"))
SIM_MIN = float(os.getenv("SIM_MIN", "0.35"))

SYSTEM_PROMPT = (
    "You are a calm, precise assistant. Keep answers concise but useful. "
    "Ask brief clarifying questions when needed."
)

EXTRACT_PROMPT = (
    "Extract candidate atomic memories from the exchange. Each memory must be one sentence, "
    "stable, and future-useful. Focus on user preferences, goals, projects, decisions, "
    "or stable facts. Return JSON only:\n"
    '{ "memories": ["...","..."] }\n'
    "If there are no good memories, return {\"memories\": []}.\n\n"
    "Exchange:\n"
    "User: {user}\n"
    "Assistant: {assistant}"
)

INDEX_HTML = """<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Titan Memory</title>
    <style>
      :root {
        --bg-1: #0b0f1a;
        --bg-2: #0f1b2e;
        --panel: rgba(12, 18, 30, 0.88);
        --border: #1d2a3b;
        --accent: #4dd0b6;
        --accent-2: #7aa2ff;
        --text: #eef7f6;
        --muted: #9fb1b0;
        --user: rgba(77, 208, 182, 0.18);
        --assistant: rgba(122, 162, 255, 0.16);
        --shadow: 0 16px 40px rgba(0, 0, 0, 0.35);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        min-height: 100vh;
        color: var(--text);
        background:
          radial-gradient(1200px 800px at 20% -10%, rgba(77, 208, 182, 0.18), transparent 60%),
          radial-gradient(1000px 700px at 90% 10%, rgba(122, 162, 255, 0.16), transparent 55%),
          linear-gradient(160deg, var(--bg-1), var(--bg-2));
        font-family: "Avenir Next", "Futura", "Gill Sans", "Helvetica Neue", sans-serif;
        letter-spacing: 0.2px;
      }
      .shell {
        display: flex;
        min-height: 100vh;
      }
      .chat {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 18px;
        padding: 32px 28px 24px 28px;
      }
      header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 12px;
        padding-bottom: 8px;
      }
      .brand {
        font-size: 18px;
        letter-spacing: 2px;
        text-transform: uppercase;
        color: var(--accent);
      }
      .subtle {
        font-size: 12px;
        color: var(--muted);
      }
      .messages {
        flex: 1;
        display: flex;
        flex-direction: column;
        gap: 14px;
        padding: 8px 6px 16px 6px;
        overflow-y: auto;
      }
      .message {
        max-width: 72ch;
        padding: 12px 16px;
        border-radius: 16px;
        border: 1px solid var(--border);
        background: var(--panel);
        box-shadow: var(--shadow);
        animation: rise 0.2s ease-out;
        line-height: 1.5;
        font-size: 15px;
      }
      .message.user {
        align-self: flex-end;
        background: var(--user);
        border-color: rgba(77, 208, 182, 0.35);
      }
      .message.assistant {
        align-self: flex-start;
        background: var(--assistant);
        border-color: rgba(122, 162, 255, 0.35);
      }
      .message.typing {
        color: var(--muted);
        font-style: italic;
      }
      .composer {
        display: flex;
        gap: 12px;
        padding: 16px;
        border-radius: 18px;
        background: rgba(10, 15, 28, 0.8);
        border: 1px solid var(--border);
        box-shadow: var(--shadow);
      }
      .composer textarea {
        flex: 1;
        resize: none;
        border: none;
        outline: none;
        background: transparent;
        color: var(--text);
        font-size: 15px;
        min-height: 48px;
        font-family: inherit;
      }
      .composer button {
        border: none;
        background: linear-gradient(120deg, var(--accent), var(--accent-2));
        color: #08131a;
        font-weight: 600;
        padding: 12px 18px;
        border-radius: 12px;
        cursor: pointer;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
      }
      .composer button:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 18px rgba(77, 208, 182, 0.2);
      }
      .memory {
        width: 320px;
        border-left: 1px solid var(--border);
        background: rgba(8, 12, 24, 0.9);
        padding: 28px 22px;
        display: flex;
        flex-direction: column;
        gap: 18px;
      }
      .memory.hidden {
        display: none;
      }
      .memory-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 10px;
      }
      .memory-title {
        font-size: 14px;
        letter-spacing: 1.6px;
        text-transform: uppercase;
        color: var(--accent-2);
      }
      .memory-status {
        font-size: 12px;
        color: var(--muted);
      }
      .memory-list {
        display: flex;
        flex-direction: column;
        gap: 12px;
      }
      .memory-item {
        border-left: 2px solid rgba(122, 162, 255, 0.4);
        padding-left: 10px;
        font-size: 13px;
        line-height: 1.4;
        color: #d7e3e2;
      }
      .memory-footer {
        font-size: 12px;
        color: var(--muted);
        display: flex;
        align-items: center;
        justify-content: space-between;
      }
      .ghost {
        border: 1px solid var(--border);
        background: transparent;
        color: var(--text);
        padding: 6px 10px;
        border-radius: 10px;
        font-size: 12px;
        cursor: pointer;
      }
      a.link {
        color: var(--accent);
        text-decoration: none;
      }
      a.link:hover {
        text-decoration: underline;
      }
      @keyframes rise {
        from { opacity: 0; transform: translateY(6px); }
        to { opacity: 1; transform: translateY(0); }
      }
      @media (max-width: 900px) {
        .shell { flex-direction: column; }
        .memory { width: 100%; border-left: none; border-top: 1px solid var(--border); }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <section class="chat">
        <header>
          <div>
            <div class="brand">Titan Memory</div>
            <div class="subtle">Local-first chat + atomic memory</div>
          </div>
          <button class="ghost" id="toggleMemory">Memory</button>
        </header>
        <div class="messages" id="messages"></div>
        <form class="composer" id="composer">
          <textarea id="input" placeholder="Say what matters..."></textarea>
          <button type="submit" id="send">Send</button>
        </form>
      </section>
      <aside class="memory hidden" id="memoryPane">
        <div class="memory-header">
          <div class="memory-title">Recent Memories</div>
          <div class="memory-status" id="memoryStatus">idle</div>
        </div>
        <div class="memory-list" id="memoryList"></div>
        <div class="memory-footer">
          <span id="memoryCount">0</span>
          <a class="link" href="/graph" target="_blank">Open graph</a>
        </div>
      </aside>
    </div>
    <script>
      const messagesEl = document.getElementById("messages");
      const inputEl = document.getElementById("input");
      const composerEl = document.getElementById("composer");
      const toggleMemoryEl = document.getElementById("toggleMemory");
      const memoryPaneEl = document.getElementById("memoryPane");
      const memoryListEl = document.getElementById("memoryList");
      const memoryStatusEl = document.getElementById("memoryStatus");
      const memoryCountEl = document.getElementById("memoryCount");

      let sessionId = localStorage.getItem("titan_session_id");
      let memoryCount = 0;

      function scrollToBottom() {
        messagesEl.scrollTop = messagesEl.scrollHeight;
      }

      function addMessage(role, text, typing = false) {
        const el = document.createElement("div");
        el.className = `message ${role}`;
        if (typing) el.classList.add("typing");
        el.textContent = text;
        messagesEl.appendChild(el);
        scrollToBottom();
        return el;
      }

      async function ensureSession() {
        if (sessionId) return sessionId;
        const res = await fetch("/api/session");
        const data = await res.json();
        sessionId = data.session_id;
        localStorage.setItem("titan_session_id", sessionId);
        return sessionId;
      }

      async function loadHistory() {
        if (!sessionId) return;
        const res = await fetch(`/api/history?session_id=${sessionId}`);
        if (!res.ok) return;
        const data = await res.json();
        messagesEl.innerHTML = "";
        data.messages.forEach((msg) => addMessage(msg.role, msg.content));
      }

      async function refreshMemories() {
        if (!sessionId) return;
        const res = await fetch(`/api/memories?session_id=${sessionId}&limit=6`);
        if (!res.ok) return;
        const data = await res.json();
        memoryListEl.innerHTML = "";
        data.memories.forEach((mem) => {
          const item = document.createElement("div");
          item.className = "memory-item";
          item.textContent = mem.text;
          memoryListEl.appendChild(item);
        });
        memoryCountEl.textContent = data.count.toString();
        if (data.count > memoryCount) {
          memoryStatusEl.textContent = "updated";
        } else {
          memoryStatusEl.textContent = "idle";
        }
        memoryCount = data.count;
      }

      async function sendMessage(text) {
        await ensureSession();
        addMessage("user", text);
        const typingEl = addMessage("assistant", "Thinking...", true);
        inputEl.value = "";
        const res = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ session_id: sessionId, message: text })
        });
        const data = await res.json();
        typingEl.remove();
        addMessage("assistant", data.assistant);
        setTimeout(refreshMemories, 800);
      }

      composerEl.addEventListener("submit", async (event) => {
        event.preventDefault();
        const text = inputEl.value.trim();
        if (!text) return;
        inputEl.focus();
        await sendMessage(text);
      });

      toggleMemoryEl.addEventListener("click", () => {
        memoryPaneEl.classList.toggle("hidden");
      });

      inputEl.addEventListener("keydown", (event) => {
        if (event.key === "Enter" && !event.shiftKey) {
          event.preventDefault();
          composerEl.requestSubmit();
        }
      });

      (async function init() {
        await ensureSession();
        await loadHistory();
        await refreshMemories();
      })();
    </script>
  </body>
</html>
"""

app = FastAPI()


class ChatRequest(BaseModel):
    session_id: str
    message: str


class ChatAdapter:
    def chat(self, messages: List[Dict[str, str]], format_hint: Optional[str] = None) -> str:
        raise NotImplementedError


class OllamaChatAdapter(ChatAdapter):
    def __init__(self, model: str, base_url: str) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def chat(self, messages: List[Dict[str, str]], format_hint: Optional[str] = None) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if format_hint:
            payload["format"] = format_hint
        try:
            r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
            r.raise_for_status()
        except requests.HTTPError:
            if format_hint:
                payload.pop("format", None)
                r = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=120)
                r.raise_for_status()
            else:
                raise
        data = r.json()
        return data["message"]["content"]


def get_adapter() -> ChatAdapter:
    if CHAT_BACKEND != "ollama":
        raise HTTPException(status_code=400, detail="Only ollama backend is supported in MVP.")
    return OllamaChatAdapter(model=CHAT_MODEL, base_url=OLLAMA_BASE_URL)


def get_extract_adapter() -> ChatAdapter:
    return OllamaChatAdapter(model=EXTRACT_MODEL, base_url=OLLAMA_BASE_URL)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def session_path(session_id: str) -> Path:
    return SESSIONS_DIR / f"{session_id}.json"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    SESSIONS_DIR.mkdir(exist_ok=True)


def load_session(session_id: str) -> Dict[str, Any]:
    path = session_path(session_id)
    if path.exists():
        return read_json(path, {"id": session_id, "created_at": now_iso(), "messages": []})
    session = {"id": session_id, "created_at": now_iso(), "messages": []}
    write_json(path, session)
    return session


def save_session(session: Dict[str, Any]) -> None:
    write_json(session_path(session["id"]), session)


def next_turn(session: Dict[str, Any]) -> int:
    return sum(1 for msg in session["messages"] if msg["role"] == "user") + 1


def sanitize_memories(memories: List[str]) -> List[str]:
    cleaned = []
    seen = set()
    for mem in memories:
        text = re.sub(r"^[\\-\\d\\.\\s]+", "", str(mem)).strip()
        if not text or len(text) < 6:
            continue
        if text in seen:
            continue
        cleaned.append(text)
        seen.add(text)
    return cleaned


def fallback_memories(user_text: str) -> List[str]:
    parts = re.split(r"(?<=[.!?])\\s+", user_text.strip())
    return sanitize_memories(parts[:3])


def extract_atomic_memories(user_text: str, assistant_text: str) -> List[str]:
    adapter = get_extract_adapter()
    prompt = EXTRACT_PROMPT.format(user=user_text.strip(), assistant=assistant_text.strip())
    content = adapter.chat(
        [{"role": "system", "content": "You output strict JSON only."},
         {"role": "user", "content": prompt}],
        format_hint="json",
    )
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        return fallback_memories(user_text)

    if isinstance(data, list):
        return sanitize_memories([str(x) for x in data])

    memories = data.get("memories") if isinstance(data, dict) else None
    if not isinstance(memories, list):
        return fallback_memories(user_text)
    return sanitize_memories([str(x) for x in memories])


def append_memories(session_id: str, turn: int, user_text: str, assistant_text: str) -> List[Dict[str, Any]]:
    extracted = extract_atomic_memories(user_text, assistant_text)
    if not extracted:
        return []
    ts = now_iso()
    records = []
    for idx, text in enumerate(extracted):
        records.append(
            {
                "id": f"{session_id}:{turn}:{idx}",
                "text": text,
                "ts": ts,
                "session_id": session_id,
                "turn": turn,
                "provenance": {"user": user_text, "assistant": assistant_text},
            }
        )
    all_memories = read_json(MEMORIES_FILE, [])
    all_memories.extend(records)
    write_json(MEMORIES_FILE, all_memories)
    return records


def build_graph(all_memories: List[Dict[str, Any]]) -> None:
    if not all_memories:
        return

    facts = [m["text"] for m in all_memories]
    vectors = ollama_embed(facts, model=EMBED_MODEL, base_url=OLLAMA_BASE_URL)
    edges = build_similarity_edges(vectors, top_k=SIM_TOP_K, min_sim=SIM_MIN)

    graph = nx.Graph()
    session_docs = {}
    for mem in all_memories:
        sid = mem["session_id"]
        doc_id = session_docs.get(sid)
        if not doc_id:
            doc_id = f"doc:{sid}"
            session_docs[sid] = doc_id
            label = f"S {sid[:6]}"
            graph.add_node(doc_id, label=label, title=f"Session {sid}", kind="document")

    for idx, mem in enumerate(all_memories):
        mem_id = mem["id"]
        graph.add_node(
            mem_id,
            label=f"M{idx}",
            title=mem["text"],
            kind="memory",
            session_id=mem["session_id"],
        )
        graph.add_edge(session_docs[mem["session_id"]], mem_id, weight=1.0, kind="source")

    for a, b, w in edges:
        graph.add_edge(all_memories[a]["id"], all_memories[b]["id"], weight=w, kind="similarity")

    net = Network(height="700px", width="100%", directed=False)
    net.from_nx(graph)

    for edge in net.edges:
        if edge.get("kind") == "similarity":
            edge["title"] = f"cosine_sim={edge.get('weight', 0):.3f}"
            edge["value"] = max(1.0, 10.0 * float(edge.get("weight", 0)))

    html = net.generate_html()
    (OUT_DIR / "graph.html").write_text(html, encoding="utf-8")

    artifacts = {
        "model": EMBED_MODEL,
        "facts": facts,
        "embedding_dim": int(vectors[0].shape[0]),
        "memory_count": len(facts),
    }
    write_json(OUT_DIR / "artifacts.json", artifacts)


def run_memory_pipeline(session_id: str, turn: int, user_text: str, assistant_text: str) -> None:
    new_records = append_memories(session_id, turn, user_text, assistant_text)
    if not new_records:
        return
    all_memories = read_json(MEMORIES_FILE, [])
    build_graph(all_memories)


@app.get("/")
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/graph")
def graph() -> FileResponse:
    path = OUT_DIR / "graph.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Graph not generated yet.")
    return FileResponse(path)


@app.get("/api/session")
def create_session() -> Dict[str, str]:
    ensure_dirs()
    session_id = uuid.uuid4().hex
    session = {"id": session_id, "created_at": now_iso(), "messages": []}
    write_json(session_path(session_id), session)
    return {"session_id": session_id}


@app.get("/api/history")
def get_history(session_id: str) -> Dict[str, Any]:
    ensure_dirs()
    session = load_session(session_id)
    return {"messages": session["messages"]}


@app.get("/api/memories")
def get_memories(session_id: Optional[str] = None, limit: int = 8) -> Dict[str, Any]:
    ensure_dirs()
    all_memories = read_json(MEMORIES_FILE, [])
    if session_id:
        all_memories = [m for m in all_memories if m["session_id"] == session_id]
    sliced = all_memories[-limit:]
    return {"memories": sliced[::-1], "count": len(all_memories)}


@app.post("/api/chat")
def chat(req: ChatRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    ensure_dirs()
    session = load_session(req.session_id)
    user_text = req.message.strip()
    if not user_text:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    turn = next_turn(session)
    chat_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + [
        {"role": msg["role"], "content": msg["content"]} for msg in session["messages"]
    ]
    chat_messages.append({"role": "user", "content": user_text})

    adapter = get_adapter()
    assistant_text = adapter.chat(chat_messages)

    session["messages"].append({"role": "user", "content": user_text, "ts": now_iso(), "turn": turn})
    session["messages"].append(
        {"role": "assistant", "content": assistant_text, "ts": now_iso(), "turn": turn}
    )
    save_session(session)

    background_tasks.add_task(run_memory_pipeline, session["id"], turn, user_text, assistant_text)

    return {
        "session_id": session["id"],
        "turn": turn,
        "assistant": assistant_text,
        "memory_status": "queued",
    }


ensure_dirs()

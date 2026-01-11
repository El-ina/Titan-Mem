import json
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
from pyvis.network import Network

from embedding.embedder import embed
from knowledge_graph.similarity import build_similarity_edges
from storage.sessions import OUT_DIR


MEMORIES_FILE = OUT_DIR / "memories.json"
GRAPH_FILE = OUT_DIR / "graph.html"
ARTIFACTS_FILE = OUT_DIR / "artifacts.json"


def load_memories() -> List[Dict[str, Any]]:
    if not MEMORIES_FILE.exists():
        return []

    with open(MEMORIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(sim_top_k: int = 2, sim_min: float = 0.35) -> None:
    all_memories = load_memories()

    if not all_memories:
        return

    facts = [m["text"] for m in all_memories]
    vectors = embed(facts)
    edges = build_similarity_edges(vectors, top_k=sim_top_k, min_sim=sim_min)

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
    net.show_buttons(filter_=["physics"])
    net.from_nx(graph)

    for edge in net.edges:
        if edge.get("kind") == "similarity":
            edge["title"] = f"cosine_sim={edge.get('weight', 0):.3f}"
            edge["value"] = max(1.0, 10.0 * float(edge.get("weight", 0)))

    html = net.generate_html()
    
    lines = html.split('\n')
    new_lines = []
    for line in lines:
        if '"enabled": false' in line:
            new_lines.append(line.replace('false', 'true'))
            new_lines.append('        "filter": ["physics"],')
        else:
            new_lines.append(line)
    
    GRAPH_FILE.write_text('\n'.join(new_lines), encoding="utf-8")

    artifacts = {
        "model": "auto",
        "facts": facts,
        "embedding_dim": int(vectors[0].shape[0]),
        "memory_count": len(facts),
    }

    with open(ARTIFACTS_FILE, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)

import json
import math
import requests
from pathlib import Path

import numpy as np
import networkx as nx
from pyvis.network import Network

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "nomic-embed-text:v1.5"

HARDCODED_FACTS = [
    "User wants a local tool that converts past AI conversations into structured memory.",
    "Pipeline will extract atomic facts, embed them, and build a knowledge graph.",
    "For MVP, skip extraction and test embeddings + graphing with hardcoded facts.",
    "Embeddings are produced by a local model and used for semantic similarity.",
    "Graph edges connect facts that are close in embedding space.",
    "Later, plug in an LLM fact extractor to make it end-to-end.",
]

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def ollama_embed(texts, model=MODEL, base_url=OLLAMA_BASE_URL):
    """
    Uses Ollama /api/embed (recommended). Falls back to /api/embeddings if needed.
    The response shape can vary by version, so we normalize it.
    """
    # Try /api/embed
    r = requests.post(
        f"{base_url}/api/embed",
        json={"model": model, "input": texts},
        timeout=120,
    )
    if r.status_code == 404:
        # Older endpoint
        r = requests.post(
            f"{base_url}/api/embeddings",
            json={"model": model, "input": texts},
            timeout=120,
        )
    r.raise_for_status()
    data = r.json()

    # Normalize possible response shapes:
    # - {"embeddings": [float, ...]} (single)
    # - {"embeddings": [[float, ...], ...]} (batch)
    # - {"embedding": [float, ...]} (older)
    if "embeddings" in data:
        emb = data["embeddings"]
        if isinstance(emb, list) and emb and isinstance(emb[0], (int, float)):
            return [np.array(emb, dtype=np.float32)]
        return [np.array(v, dtype=np.float32) for v in emb]
    if "embedding" in data:
        return [np.array(data["embedding"], dtype=np.float32)]
    raise ValueError(f"Unexpected embedding response keys: {list(data.keys())}")

def build_similarity_edges(vectors, top_k=2, min_sim=0.35):
    n = len(vectors)
    edges = []
    for i in range(n):
        sims = []
        for j in range(n):
            if i == j:
                continue
            sim = cosine_similarity(vectors[i], vectors[j])
            sims.append((j, sim))
        sims.sort(key=lambda x: x[1], reverse=True)
        for j, sim in sims[:top_k]:
            if sim >= min_sim:
                # store undirected edge with weight
                a, b = sorted((i, j))
                edges.append((a, b, sim))
    # dedupe by keeping max weight for same pair
    best = {}
    for a, b, w in edges:
        best[(a, b)] = max(best.get((a, b), 0.0), w)
    return [(a, b, w) for (a, b), w in best.items()]

def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    vectors = ollama_embed(HARDCODED_FACTS)

    # Save raw artifacts so we can inspect/debug
    artifacts = {
        "model": MODEL,
        "facts": HARDCODED_FACTS,
        "embedding_dim": int(vectors[0].shape[0]),
    }
    (out_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")

    # Build graph
    G = nx.Graph()

    # Optional: add a "source document" node (nice habit for later)
    doc_id = "doc:hardcoded"
    G.add_node(doc_id, label="Hardcoded Facts", title="Source: hardcoded test set", kind="document")

    for idx, fact in enumerate(HARDCODED_FACTS):
        node_id = f"fact:{idx}"
        G.add_node(
            node_id,
            label=f"F{idx}",
            title=fact,
            kind="fact",
            text=fact,
            provenance=doc_id,
        )
        G.add_edge(doc_id, node_id, weight=1.0, kind="source")

    edges = build_similarity_edges(vectors, top_k=2, min_sim=0.35)
    for a, b, w in edges:
        G.add_edge(f"fact:{a}", f"fact:{b}", weight=w, kind="similarity")

    # Visualize with PyVis (NetworkX integration)
    net = Network(height="700px", width="100%", directed=False)
    net.from_nx(G)

    # Make similarity edges stand out via title/weight
    for e in net.edges:
        if e.get("kind") == "similarity":
            e["title"] = f"cosine_sim={e.get('weight', 0):.3f}"
            # PyVis uses "value" for edge thickness scaling
            e["value"] = max(1.0, 10.0 * float(e.get("weight", 0)))

    net.show_buttons(filter_=["physics"])
    html = net.generate_html()
    (out_dir / "graph.html").write_text(html, encoding="utf-8")

    print("Wrote:")
    print(" - out/artifacts.json")
    print(" - out/graph.html (open in browser)")

if __name__ == "__main__":
    main()

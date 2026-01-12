import json
from pathlib import Path
from typing import Any, Dict, List

import networkx as nx
from pyvis.network import Network
from networkx.algorithms.community import greedy_modularity_communities

from embedding.embedder import embed
from knowledge_graph.similarity import build_similarity_edges
from storage.sessions import OUT_DIR


MEMORIES_FILE = OUT_DIR / "memories.json"
GRAPH_FILE = OUT_DIR / "graph.html"
ARTIFACTS_FILE = OUT_DIR / "artifacts.json"


PALETTE = [
    "#22c55e",  # green
    "#38bdf8",  # sky
    "#a78bfa",  # purple
    "#fbbf24",  # amber
    "#f472b6",  # pink
    "#34d399",  # emerald
    "#60a5fa",  # blue
]


def load_memories() -> List[Dict[str, Any]]:
    if not MEMORIES_FILE.exists():
        return []

    with open(MEMORIES_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_graph(sim_top_k: int = 2, sim_min: float = 0.35) -> None:
    all_memories = load_memories()

    if not all_memories:
        # Don't create empty graph - just return
        return

    facts = [m["text"] for m in all_memories]
    vectors = embed(facts)
    edges = build_similarity_edges(vectors, top_k=sim_top_k, min_sim=sim_min)
    sim_weights = [w for _, _, w in edges] or [0.0]
    w_min, w_max = min(sim_weights), max(sim_weights)

    graph = nx.Graph()

    # Group memories by session
    session_docs = {}
    for mem in all_memories:
        sid = mem["session_id"]
        doc_id = session_docs.get(sid)
        if not doc_id:
            doc_id = f"doc:{sid}"
            session_docs[sid] = doc_id
            label = f"S {sid[:6]}"
            graph.add_node(doc_id, label=label, title=f"Session {sid}", kind="document")

    # Add memory nodes
    for idx, mem in enumerate(all_memories):
        mem_id = mem["id"]
        graph.add_node(
            mem_id,
            label=f"M{idx}",
            title=mem["text"],
            kind="memory",
            session_id=mem["session_id"],
        )
        # Connect memory to its session
        graph.add_edge(session_docs[mem["session_id"]], mem_id, weight=1.0, kind="source")

    # Add similarity edges between memories
    for a, b, w in edges:
        graph.add_edge(all_memories[a]["id"], all_memories[b]["id"], weight=w, kind="similarity")

    # Community detection on memory-only subgraph for color coding
    memory_ids = [m["id"] for m in all_memories]
    memory_sub = graph.subgraph(memory_ids).copy()
    communities = list(greedy_modularity_communities(memory_sub)) if memory_sub.number_of_edges() else []
    node_to_group = {}
    for gi, comm in enumerate(communities):
        for n in comm:
            node_to_group[n] = gi

    # Create pyvis network with polished style - full viewport
    net = Network(height="100vh", width="100%", bgcolor="#0a0a0a", directed=False, cdn_resources="in_line")
    
    net.from_nx(graph)

    # Calculate max degree for node sizing
    degrees = dict(memory_sub.degree())
    max_deg = max(degrees.values()) if degrees else 1

    # Style memory nodes with community colors
    for node in net.nodes:
        if node.get("kind") == "memory":
            group_idx = node_to_group.get(node["id"], 0) % len(PALETTE)
            node["color"] = PALETTE[group_idx]
            node["group"] = group_idx
            # Adjust node size based on degree
            deg = degrees.get(node["id"], 0)
            node["size"] = 8 + 12 * (deg / max_deg if max_deg > 0 else 1)
        elif node.get("kind") == "document":
            node["color"] = "#2d3748"
            node["size"] = 20

    # Style edges with contrasting colors against black background
    # Filter out zero-similarity edges
    edges_to_remove = []
    for edge in net.edges:
        if edge.get("kind") == "similarity":
            # Pyvis converts 'weight' to 'width', so read from there
            w = float(edge.get("width", 0.0) or 0.0)
            # Only show connections with positive similarity
            if w <= 0:
                edges_to_remove.append(edge)
                continue

            # Thickness directly proportional to similarity (1.5-6.0 range)
            edge["width"] = 1.5 + 4.5 * w
            # Bright cyan/teal gradient - highly visible on black
            r = int(80 + 100 * w)
            g = int(200 + 55 * w)
            b = int(220 + 35 * w)
            edge["color"] = f"#{r:02x}{g:02x}{b:02x}"
            edge["title"] = f"similarity: {w:.3f}"
        elif edge.get("kind") == "source":
            # Brighter gray for source edges
            edge["color"] = "#8899aa"
            edge["width"] = 1.5

    # Remove zero-similarity edges
    for edge in edges_to_remove:
        net.edges.remove(edge)

    # Apply polished visualization options
    options = {
      "layout": {
        "improvedLayout": True
      },
      "interaction": {
        "hover": True,
        "tooltipDelay": 20,
        "navigationButtons": True,
        "keyboard": True,
        "hoverConnectedEdges": True
      },
      "nodes": {
        "shape": "circle",
        "borderWidth": 2,
        "font": { 
            "size": 14, 
            "color": "#ffffff",
            "face": "Inter, -apple-system, sans-serif"
        },
        "shadow": {
            "enabled": True,
            "color": "rgba(0,0,0,0.3)",
            "size": 5
        }
      },
      "edges": {
        "smooth": { 
            "enabled": True,
            "type": "continuous",
            "roundness": 0.5
        },
        "shadow": { "enabled": False },
        "selectionWidth": 3,
        "hoverWidth": 2
      },
      "physics": {
        "enabled": True,
        "solver": "forceAtlas2Based",
        "forceAtlas2Based": {
          "gravitationalConstant": -30,
          "centralGravity": 0.005,
          "springLength": 80,
          "springConstant": 0.06,
          "damping": 0.3,
          "avoidOverlap": 0.8
        },
        "stabilization": { "iterations": 200 }
      }
    }
    net.set_options(json.dumps(options))

    # Generate and save HTML
    html = net.generate_html()

    # Inject CSS to make entire page black with no borders, full viewport
    dark_css = """
    <style>
      html, body {
        margin: 0;
        padding: 0;
        width: 100%;
        height: 100%;
        background-color: #0a0a0a !important;
        overflow: hidden;
        border: none !important;
      }
      #mynetwork {
        width: 100% !important;
        height: 100vh !important;
        background-color: #0a0a0a !important;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
      }
      .card {
        border: none !important;
        background-color: #0a0a0a !important;
        height: 100% !important;
      }
      canvas {
        outline: none !important;
      }
    </style>
    """
    html = html.replace("<head>", "<head>" + dark_css)

    GRAPH_FILE.write_text(html, encoding="utf-8")

    # Save artifacts
    artifacts = {
        "model": "auto",
        "facts": facts,
        "embedding_dim": int(vectors[0].shape[0]),
        "memory_count": len(facts),
    }

    with open(ARTIFACTS_FILE, "w", encoding="utf-8") as f:
        json.dump(artifacts, f, indent=2)
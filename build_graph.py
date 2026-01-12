import json
from pathlib import Path

import numpy as np
import requests
import networkx as nx
from pyvis.network import Network
from networkx.algorithms.community import greedy_modularity_communities

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "nomic-embed-text:v1.5"



# -------- Embedding + similarity --------

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def ollama_embed(texts, model=MODEL, base_url=OLLAMA_BASE_URL):
    # Preferred endpoint
    r = requests.post(f"{base_url}/api/embed", json={"model": model, "input": texts}, timeout=120)
    if r.status_code == 404:
        # Older endpoint fallback
        r = requests.post(f"{base_url}/api/embeddings", json={"model": model, "input": texts}, timeout=120)
    r.raise_for_status()
    data = r.json()

    if "embeddings" in data:
        emb = data["embeddings"]
        if emb and isinstance(emb[0], (int, float)):
            return [np.array(emb, dtype=np.float32)]
        return [np.array(v, dtype=np.float32) for v in emb]
    if "embedding" in data:
        return [np.array(data["embedding"], dtype=np.float32)]
    raise ValueError(f"Unexpected embedding response keys: {list(data.keys())}")

def build_similarity_edges(vectors, top_k=3, min_sim=0.35):
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
                a, b = sorted((i, j))
                edges.append((a, b, sim))

    # de-dupe: keep max similarity per pair
    best = {}
    for a, b, w in edges:
        best[(a, b)] = max(best.get((a, b), 0.0), w)
    return [(a, b, w) for (a, b), w in best.items()]

# -------- Load memories --------

def load_memories():
    mem_file = Path("out/memories.json")
    if not mem_file.exists():
        return []
    with open(mem_file, "r", encoding="utf-8") as f:
        return json.load(f)

# -------- Visual style helpers --------

PALETTE = [
    "#22c55e",  # green
    "#38bdf8",  # sky
    "#a78bfa",  # purple
    "#fbbf24",  # amber
    "#f472b6",  # pink
    "#34d399",  # emerald
    "#60a5fa",  # blue
]

def rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"

def main():
    out_dir = Path("out")
    out_dir.mkdir(exist_ok=True)

    memories = load_memories()
    
    if not memories:
        print("No memories found in out/memories.json")
        return

    facts = [m["text"] for m in memories]
    vectors = ollama_embed(facts)
    emb_dim = int(vectors[0].shape[0])

    # Similarity edges between facts
    sim_edges = build_similarity_edges(vectors, top_k=3, min_sim=0.35)
    sim_weights = [w for _, _, w in sim_edges] or [0.0]
    w_min, w_max = min(sim_weights), max(sim_weights)

    # Build a simple graph object for community + degrees
    G = nx.Graph()

    # Group memories by session
    session_docs = {}
    for mem in memories:
        sid = mem["session_id"]
        doc_id = session_docs.get(sid)
        if not doc_id:
            doc_id = f"doc:{sid}"
            session_docs[sid] = doc_id
            label = f"S {sid[:6]}"
            G.add_node(doc_id, label=label, title=f"Session {sid}", kind="document")

    fact_ids = []
    for i, mem in enumerate(memories):
        nid = mem["id"]
        fact_ids.append(nid)
        G.add_node(nid, kind="memory", title=mem["text"])

        # Connect memory to its session
        G.add_edge(session_docs[mem["session_id"]], nid, kind="source", weight=1.0)

    for a, b, w in sim_edges:
        G.add_edge(memories[a]["id"], memories[b]["id"], kind="similarity", weight=float(w))

    # Community detection on the memory-only subgraph (for cluster colors)
    fact_sub = G.subgraph(fact_ids).copy()
    communities = list(greedy_modularity_communities(fact_sub)) if fact_sub.number_of_edges() else []
    node_to_group = {}
    for gi, comm in enumerate(communities):
        for n in comm:
            node_to_group[n] = gi

    # Create a clean neural network style visualization
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#0a0a0a",
        directed=False,
        cdn_resources="in_line",
    )

    # Add document nodes (sessions)
    for sid, doc_id in session_docs.items():
        label = f"S {sid[:6]}"
        net.add_node(
            doc_id,
            label=label,
            title=f"Session {sid}",
            size=20,
            color="#2d3748",
        )

    # Add memory nodes (colored by community/knowledge area)
    degrees = dict(fact_sub.degree()) if fact_sub.number_of_nodes() else {}
    max_deg = max(degrees.values()) if degrees else 1

    for nid in fact_ids:
        deg = degrees.get(nid, 0)
        size = 8 + (12 * (deg / max_deg))  # larger nodes (8-20 range)
        
        # Color by community/knowledge area
        group_idx = node_to_group.get(nid, 0) % len(PALETTE)
        node_color = PALETTE[group_idx]

        net.add_node(
            nid,
            label="",  # clean look
            title=G.nodes[nid]["title"],
            size=size,
            color=node_color,
            group=group_idx,
        )

    # Add edges: polished with thickness based on similarity strength
    for u, v, data in G.edges(data=True):
        kind = data.get("kind")
        w = float(data.get("weight", 0.0))

        if kind == "source":
            net.add_edge(
                u, v,
                width=1,
                color="#4a5568",
                title="source",
                smooth=True,
            )
        else:
            # map similarity to thickness (thicker for stronger connections)
            if w_max > w_min:
                t = (w - w_min) / (w_max - w_min)
            else:
                t = 0.5

            width = 1.0 + 4.0 * t  # 1.0-5.0 range for visible thickness
            
            # Color gradient from light blue to dark blue based on strength
            base = 100 + 155 * t
            color = f"#{int(base):02x}{int(base):02x}ff"
            hover_color = f"#{int(base+30):02x}{int(base+30):02x}ff"

            net.add_edge(
                u, v,
                width=width,
                color={
                    "color": color,
                    "hover": hover_color,
                    "highlight": hover_color
                },
                title=f"similarity: {w:.3f}",
                smooth=True,
            )

    # Knowledge graph style options - polished diagram look
    options = {
      "layout": {
        "improvedLayout": True,
        "hierarchical": {
          "enabled": False,
          "sortMethod": "hubsize"
        }
      },

      "interaction": {
        "hover": True,
        "tooltipDelay": 20,
        "navigationButtons": True,
        "keyboard": True,
        "multiselect": False,
        "hoverConnectedEdges": True
      },

      "nodes": {
        "shape": "circle",
        "borderWidth": 2,
        "borderWidthSelected": 3,
        "font": { 
            "size": 14, 
            "color": "#ffffff",
            "face": "Inter, -apple-system, BlinkMacSystemFont, sans-serif",
            "strokeWidth": 2,
            "strokeColor": "#000000"
        },
        "shadow": {
            "enabled": True,
            "color": "rgba(0,0,0,0.3)",
            "size": 5,
            "x": 2,
            "y": 2
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

    # Generate html, then inject a slick HUD overlay + background styling + click-to-inspect
    html = net.generate_html()

    facts_count = len(fact_ids)
    doc_count = 1
    sim_count = len(sim_edges)
    total_edges = G.number_of_edges()

    # Generate community color legend
    legend_items = ""
    for i, color in enumerate(PALETTE):
        legend_items += f'<div style="display:flex;align-items:center;gap:8px;margin:4px 0;"><div style="width:16px;height:16px;background:{color};border-radius:50%;"></div><span style="font-size:11px;color:#a0aec0;">Knowledge Area {i+1}</span></div>'

    # Edge strength legend
    edge_legend = """
    <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
      <div style="width:16px;height:2px;background:#6464ff;"></div>
      <span style="font-size:11px;color:#a0aec0;">Weak</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
      <div style="width:16px;height:4px;background:#6464ff;"></div>
      <span style="font-size:11px;color:#a0aec0;">Medium</span>
    </div>
    <div style="display:flex;align-items:center;gap:8px;margin:4px 0;">
      <div style="width:16px;height:6px;background:#6464ff;"></div>
      <span style="font-size:11px;color:#a0aec0;">Strong</span>
    </div>
    """

    hud = f"""
    <style>
      html, body {{
        margin: 0;
        height: 100%;
        background: #0a0a0a;
        color: #ffffff;
        font-family: Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      }}
      #mynetwork {{
        background: #0a0a0a !important;
      }}
      .hud {{
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(10, 10, 10, 0.95);
        border: 1px solid #2d3748;
        border-radius: 8px;
        padding: 16px;
        font-size: 12px;
        max-width: 280px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.4);
        backdrop-filter: blur(10px);
      }}
      .hud .title {{
        font-weight: 700;
        margin-bottom: 12px;
        color: #ffffff;
        font-size: 14px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
      }}
      .hud .stats {{
        margin-bottom: 12px;
        color: #a0aec0;
        font-size: 11px;
        padding-bottom: 12px;
        border-bottom: 1px solid #2d3748;
      }}
      .hud .legend {{
        margin-bottom: 12px;
        padding-bottom: 12px;
        border-bottom: 1px solid #2d3748;
      }}
      .hud .legend-title {{
        font-size: 11px;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 8px;
      }}
      .hud .info {{
        color: #ffffff;
        font-size: 11px;
        line-height: 1.5;
        max-height: 120px;
        overflow: auto;
      }}
      .hud .info::-webkit-scrollbar {{
        width: 4px;
      }}
      .hud .info::-webkit-scrollbar-thumb {{
        background: #4a5568;
        border-radius: 2px;
      }}
    </style>

    <div class="hud">
      <div class="title">Titan Memory</div>
      <div class="stats">
        {facts_count} memories · {sim_count} connections
      </div>
      <div class="legend">
        <div class="legend-title">Knowledge Areas</div>
        {legend_items}
      </div>
      <div class="legend">
        <div class="legend-title">Connection Strength</div>
        {edge_legend}
      </div>
      <div class="info" id="nodeInfo">Click a node to view details</div>
    </div>
    """

    # Insert HUD right after <body>
    html = html.replace("<body>", "<body>" + hud)

    # Hook click events to show node text with better formatting
    marker = "var network = new vis.Network(container, data, options);"
    if marker in html:
        html = html.replace(
            marker,
            marker + """
            var infoEl = document.getElementById("nodeInfo");
            var nodeEl = document.querySelector(".hud .info");
            
            network.on("click", function(params) {
              if (!params.nodes || params.nodes.length === 0) {
                infoEl.textContent = "Click a node to view details";
                return;
              }
              
              var id = params.nodes[0];
              var node = data.nodes.get(id);
              if (!node) return;
              
              // Format node info
              var text = node.title || node.label || id;
              var label = node.label || "Node";
              
              infoEl.innerHTML = "<strong>" + label + "</strong><br/><br/>" + 
                text.replace(/\\n/g, "<br/>");
            });
            
            // Hover effect
            network.on("hoverNode", function(params) {
              document.body.style.cursor = "pointer";
            });
            
            network.on("blurNode", function(params) {
              document.body.style.cursor = "default";
            });
            """
        )

    # Save artifacts
    artifacts = {"model": MODEL, "facts": facts, "embedding_dim": emb_dim}
    (out_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    (out_dir / "graph.html").write_text(html, encoding="utf-8")

    print("Wrote out/graph.html and out/artifacts.json")

if __name__ == "__main__":
    main()
    # testing git flow

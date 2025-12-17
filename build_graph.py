import json
from pathlib import Path

import numpy as np
import requests
import networkx as nx
from pyvis.network import Network
from networkx.algorithms.community import greedy_modularity_communities

OLLAMA_BASE_URL = "http://localhost:11434"
MODEL = "nomic-embed-text:v1.5"

HARDCODED_FACTS = [
    "User wants a local tool that converts past AI conversations into structured memory.",
    "Pipeline will extract atomic facts, embed them, and build a knowledge graph.",
    "For MVP, skip extraction and test embeddings + graphing with hardcoded facts.",
    "Embeddings are produced by a local model and used for semantic similarity.",
    "Graph edges connect facts that are close in embedding space.",
    "Later, plug in an LLM fact extractor to make it end-to-end.",
    "The user is attempting to understand intelligence not just as computation, but as something that requires historical context to produce meaning.",
    "The user distinguishes between raw language model training data and personally meaningful data curated through intentional reflection.",
    "The user believes that meaning emerges when information is connected to a coherent past rather than treated as isolated tokens.",
    "The user is exploring whether next-token prediction in LLMs has an analogue in predicting future human actions or life events.",
    "The user conceptualizes personal atomic memories as an event stream that could be modeled similarly to language sequences.",
    "The user is motivated by the idea that structured memory could enable probabilistic prediction of future decisions or phases.",
    "The user is aware that without an explicit prediction objective, embeddings and graphs remain descriptive rather than generative.",
    "The user is drawn to the idea of an AI system that maintains continuity instead of resetting context every interaction.",
    "The user experiences discomfort with AI systems that feel intelligent but lack long-term memory or personal consistency.",
    "The user frames long-term memory as a prerequisite for agency, identity, and sustained reasoning.",
    "The user fears uncontrolled context injection and is concerned about overwhelming an AI with irrelevant personal history.",
    "The user intuitively understands that selective recall, not total recall, is necessary for usable intelligence.",
    "The user is reassured by the idea that memory retrieval can be constrained, compressed, and audited.",
    "The user values systems that expose which memories influenced a response, rather than opaque personalization.",
    "The user believes that long-term memory must be editable, versioned, and deletable to remain psychologically safe.",
    "The user sees a personal memory system as a way to reduce repeated self-contradiction and decision fatigue.",
    "The user expects that persistent memory would increase decision integrity across time.",
    "The user believes that most people repeatedly relearn the same lessons due to lack of continuity, not lack of intelligence.",
    "The user sees cumulative learning as a major leverage point that current tools fail to preserve.",
    "The user anticipates that structured memory could interrupt negative behavioral or cognitive loops early.",
    "The user believes that personal patterns are more valuable to surface than generic advice.",
    "The user understands that embeddings approximate semantic similarity but do not encode causal or temporal truth.",
    "The user recognizes that embeddings alone cannot reliably distinguish updates, contradictions, or negations.",
    "The user accepts that a reasoning layer is required to interpret relationships between memories correctly.",
    "The user conceptualizes LLMs as general intelligence operating systems rather than identity-bearing entities.",
    "The user imagines a future where personal memory layers act as an identity module attached to general AI models.",
    "The user is inspired by fictional examples but understands the real risks of misaligned memory systems.",
    "The user believes that memory quality directly determines whether an AI amplifies wisdom or personal dysfunction.",
    "The user sees their project as building a controllable second brain rather than an autonomous personality.",
    "The user is motivated by the long-term compounding effects of continuity, self-knowledge, and reduced cognitive leakage.",
]

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

    vectors = ollama_embed(HARDCODED_FACTS)
    emb_dim = int(vectors[0].shape[0])

    # Similarity edges between facts
    sim_edges = build_similarity_edges(vectors, top_k=3, min_sim=0.35)
    sim_weights = [w for _, _, w in sim_edges] or [0.0]
    w_min, w_max = min(sim_weights), max(sim_weights)

    # Build a simple graph object for community + degrees
    G = nx.Graph()
    doc_id = "doc:hardcoded"
    G.add_node(doc_id, kind="document", title="Hardcoded Facts")

    fact_ids = []
    for i, fact in enumerate(HARDCODED_FACTS):
        nid = f"fact:{i}"
        fact_ids.append(nid)
        G.add_node(nid, kind="fact", title=fact)

        # provenance edge (doc -> fact)
        G.add_edge(doc_id, nid, kind="source", weight=1.0)

    for a, b, w in sim_edges:
        G.add_edge(f"fact:{a}", f"fact:{b}", kind="similarity", weight=float(w))

    # Community detection on the fact-only subgraph (for cluster colors)
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
        bgcolor="#000000",
        font_color="#ffffff",
        directed=False,
        cdn_resources="in_line",
    )

    # Add document node (input layer)
    net.add_node(
        doc_id,
        label="SOURCE",
        title="Source: hardcoded test set",
        size=5,
        color="#ffff00",
    )

    # Add fact nodes (small dots, labels hidden, text on hover/click)
    degrees = dict(fact_sub.degree()) if fact_sub.number_of_nodes() else {}
    max_deg = max(degrees.values()) if degrees else 1

    for nid in fact_ids:
        deg = degrees.get(nid, 0)
        size = 3 + (3 * (deg / max_deg))  # much smaller dots (3-6 range)

        net.add_node(
            nid,
            label="",  # clean look
            title=G.nodes[nid]["title"],
            size=size,
            color="#ffff00",
        )

    # Add edges: clean straight lines
    for u, v, data in G.edges(data=True):
        kind = data.get("kind")
        w = float(data.get("weight", 0.0))

        if kind == "source":
            net.add_edge(
                u, v,
                width=0.5,
                color="#444444",
                title="provenance",
                smooth=False,
            )
        else:
            # map similarity to thickness (thinner edges)
            if w_max > w_min:
                t = (w - w_min) / (w_max - w_min)
            else:
                t = 0.5

            width = 0.5 + 1.5 * t  # 0.5-2.0 range
            brightness = int(80 + 100 * t)  # 80-180 range
            color = f"#{brightness:02x}{brightness:02x}{brightness+20:02x}"

            net.add_edge(
                u, v,
                width=width,
                color=color,
                title=f"cosine_sim={w:.3f}",
                smooth=False,
            )

    # Knowledge graph style options
    options = {
      "layout": {
        "improvedLayout": True
      },

      "interaction": {
        "hover": True,
        "tooltipDelay": 20,
        "navigationButtons": True,
        "keyboard": True
      },

      "nodes": {
        "shape": "dot",
        "borderWidth": 0,
        "font": { "size": 10, "color": "#ffffff" },
        "shadow": { "enabled": False }
      },

      "edges": {
        "smooth": { "enabled": False },
        "shadow": { "enabled": False },
        "selectionWidth": 2,
        "hoverWidth": 1
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

    hud = f"""
    <style>
      html, body {{
        margin: 0;
        height: 100%;
        background: #000000;
        color: #ffffff;
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      }}
      #mynetwork {{
        background: #000000 !important;
      }}
      .hud {{
        position: fixed;
        top: 20px;
        right: 20px;
        background: rgba(0, 0, 0, 0.8);
        border: 1px solid #333333;
        padding: 15px;
        font-size: 12px;
        max-width: 250px;
      }}
      .hud .title {{
        font-weight: bold;
        margin-bottom: 10px;
        color: #00ffff;
      }}
      .hud .stats {{
        margin-bottom: 10px;
        color: #cccccc;
      }}
      .hud .info {{
        color: #ffffff;
        border-top: 1px solid #333333;
        padding-top: 10px;
        max-height: 150px;
        overflow: auto;
      }}
    </style>

    <div class="hud">
      <div class="title">TITAN MEMORY</div>
      <div class="stats">
        {facts_count} facts · {sim_count} connections
      </div>
      <div class="info" id="nodeInfo">Click node to inspect</div>
    </div>
    """

    # Insert HUD right after <body>
    html = html.replace("<body>", "<body>" + hud)

    # Hook click events to show node text
    marker = "var network = new vis.Network(container, data, options);"
    if marker in html:
        html = html.replace(
            marker,
            marker + """
            var infoEl = document.getElementById("nodeInfo");
            network.on("click", function(params) {
              if (!params.nodes || params.nodes.length === 0) return;
              var id = params.nodes[0];
              var node = data.nodes.get(id);
              if (!node) return;
              infoEl.textContent = node.title || node.label || id;
            });
            """
        )

    # Save artifacts
    artifacts = {"model": MODEL, "facts": HARDCODED_FACTS, "embedding_dim": emb_dim}
    (out_dir / "artifacts.json").write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    (out_dir / "graph.html").write_text(html, encoding="utf-8")

    print("Wrote out/graph.html and out/artifacts.json")

if __name__ == "__main__":
    main()
    # testing git flow

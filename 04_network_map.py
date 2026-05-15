"""
Step 4: Co-occurrence Network Visualization (Map 2 — The Structure Map)

Builds a network where:
- Nodes = terms
- Edges = co-occurrence strength
- Colors = community/cluster membership

This reveals the sub-domains of HCI research.
"""

import os

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from pyvis.network import Network
from scipy import sparse

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def load_data():
    dtm = sparse.load_npz(os.path.join(DATA_DIR, "dtm.npz"))
    vocab = np.load(os.path.join(DATA_DIR, "vocab.npy"), allow_pickle=True)
    return dtm, vocab


GENERIC_STOPWORDS = {
    "study", "results", "research", "based", "using", "used", "paper",
    "approach", "method", "methods", "proposed", "present", "model",
    "data", "analysis", "system", "new", "different", "provide",
    "also", "however", "two", "three", "first", "one", "well",
    "use", "may", "can", "including", "et", "al", "work", "order",
    "show", "find", "like", "make", "way", "need", "set", "high",
    "low", "large", "number", "important", "significant", "result",
    "group", "level", "studies", "aim", "aims", "objective", "purpose",
    "abstract", "article", "review", "conclude", "conclusion",
    "discuss", "shown", "demonstrate", "suggest", "indicate",
    "investigate", "examine", "explore", "develop", "improve",
    "compare", "evaluate", "apply", "consider", "describe",
    "report", "identify", "assess", "determine", "test", "perform",
    "existing", "current", "recent", "previous", "related",
    "various", "several", "general", "specific", "particular",
    "potential", "effective", "better", "best", "good", "possible",
    "widely", "highly", "especially", "particularly", "overall",
    "mainly", "primarily", "furthermore", "moreover", "addition",
    "finally", "second", "third", "following", "called", "known",
    "total", "average", "mean", "value", "values", "sample",
    "process", "processes", "role", "impact", "effect", "effects",
    "factor", "factors", "increase", "decrease", "change", "changes",
    "problem", "problems", "solution", "challenge", "challenges",
    "issue", "issues", "key", "main", "primary", "address",
    "attention", "focus", "contribute", "present", "presents",
    "paper presents", "non", "applied", "field", "point", "view",
    "basis", "complex", "components", "strategy", "traditional",
    "context", "multiple", "term", "terms", "long", "short",
    "ability", "area", "areas", "range", "wide", "variety",
    "type", "types", "characteristics", "according", "addition",
    "furthermore", "target", "combined", "control", "active",
    "play", "crucial", "essential", "necessary", "required",
    "requires", "help", "understand", "support", "features",
}


def build_network(dtm, vocab, top_n=500, min_cooc=50):
    """
    Build a co-occurrence network from the DTM.

    Args:
        top_n: number of top terms to include (after filtering generic words)
        min_cooc: minimum co-occurrence count to create an edge
    """
    # Select top terms by document frequency, excluding generic words
    term_freq = np.asarray((dtm > 0).sum(axis=0)).flatten()

    # Filter using shared stopwords (academic methodology + non-English + non-HCI domains)
    from stopwords import is_valid_term
    valid_mask = np.array([is_valid_term(v) for v in vocab])
    filtered_freq = term_freq.copy()
    filtered_freq[~valid_mask] = 0

    top_indices = filtered_freq.argsort()[-top_n:][::-1]
    # Remove any that were zeroed out
    top_indices = top_indices[filtered_freq[top_indices] > 0]

    dtm_sub = dtm[:, top_indices]
    top_vocab = vocab[top_indices]

    # Binary presence
    binary = (dtm_sub > 0).astype(int)
    n_docs = binary.shape[0]

    # Co-occurrence matrix
    cooc = (binary.T @ binary).toarray().astype(float)
    np.fill_diagonal(cooc, 0)

    # Document frequencies per term
    doc_freq = np.asarray(binary.sum(axis=0)).flatten().astype(float)

    # Compute PMI (Pointwise Mutual Information) for edge weights
    # PMI(i,j) = log2( P(i,j) / (P(i) * P(j)) )
    # Only keep edges with positive PMI and sufficient co-occurrence
    G = nx.Graph()

    for i in range(len(top_vocab)):
        G.add_node(top_vocab[i], freq=int(term_freq[top_indices[i]]))

    for i in range(len(top_vocab)):
        for j in range(i + 1, len(top_vocab)):
            if cooc[i][j] < min_cooc:
                continue
            p_ij = cooc[i][j] / n_docs
            p_i = doc_freq[i] / n_docs
            p_j = doc_freq[j] / n_docs
            if p_i == 0 or p_j == 0:
                continue
            pmi = np.log2(p_ij / (p_i * p_j))
            if pmi > 0.5:  # only keep meaningfully associated pairs
                G.add_edge(top_vocab[i], top_vocab[j], weight=float(pmi))

    # Remove isolated nodes
    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  Removed {len(isolates)} isolated nodes")

    return G


def detect_communities(G):
    """Detect communities using Louvain method."""
    communities = nx.community.louvain_communities(G, seed=42, resolution=1.2)

    # Assign community labels to nodes
    for i, community in enumerate(communities):
        for node in community:
            G.nodes[node]["community"] = i

    print(f"Detected {len(communities)} communities")
    for i, comm in enumerate(communities):
        top_terms = sorted(comm, key=lambda n: G.nodes[n].get("freq", 0), reverse=True)[:5]
        print(f"  Community {i} ({len(comm)} terms): {', '.join(top_terms)}")

    return communities


def plot_static_network(G, communities):
    """Create a static matplotlib network visualization."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(24, 24))

    # Layout
    pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42, weight="weight")

    # Colors per community
    n_communities = len(communities)
    cmap = plt.cm.get_cmap("tab20", n_communities)
    node_colors = [cmap(G.nodes[n]["community"]) for n in G.nodes()]

    # Node sizes based on frequency
    freqs = [G.nodes[n].get("freq", 1) for n in G.nodes()]
    max_freq = max(freqs)
    node_sizes = [30 + 300 * (f / max_freq) for f in freqs]

    # Edge widths
    weights = [G[u][v]["weight"] for u, v in G.edges()]
    max_w = max(weights) if weights else 1
    edge_widths = [0.2 + 2 * (w / max_w) for w in weights]

    # Draw
    nx.draw_networkx_edges(G, pos, alpha=0.05, width=edge_widths, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax)

    # Labels for high-frequency terms only
    freq_threshold = np.percentile(freqs, 70)
    labels = {n: n for n in G.nodes() if G.nodes[n].get("freq", 0) >= freq_threshold}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, alpha=0.8, ax=ax)

    ax.set_title("HCI Research Landscape — Term Co-occurrence Network", fontsize=16)
    ax.axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "map2_network.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "map2_network.pdf"), bbox_inches="tight")
    print(f"Saved: map2_network.png/pdf")
    plt.close()


def create_interactive_network(G, communities):
    """Create a lightweight custom Canvas-based interactive HTML network."""
    import json as _json
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Pre-compute layout
    pos = nx.spring_layout(G, k=1.2, iterations=200, seed=42, weight="weight")

    # Theme labels for communities (auto-generated from top bigrams)
    theme_labels = {}
    for i, comm in enumerate(communities):
        bigrams = [n for n in comm if " " in n]
        if bigrams:
            top = sorted(bigrams, key=lambda n: G.nodes[n].get("freq", 0), reverse=True)[:2]
            theme_labels[i] = " / ".join(top).title()
        else:
            top = sorted(comm, key=lambda n: G.nodes[n].get("freq", 0), reverse=True)[:2]
            theme_labels[i] = " / ".join(top).title()

    # Saturated color palette
    colors = [
        "#FF1744", "#00E676", "#FFEA00", "#2979FF", "#FF9100",
        "#D500F9", "#00E5FF", "#F50057", "#76FF03", "#FF6D00",
        "#651FFF", "#1DE9B6", "#C6FF00", "#304FFE", "#DD2C00",
        "#00BFA5", "#FFD600", "#AA00FF", "#64DD17", "#6200EA",
    ]

    freqs = {n: G.nodes[n].get("freq", 1) for n in G.nodes()}
    max_freq = max(freqs.values())

    # For each node, get top 5 neighbors sorted by edge weight
    def get_top_neighbors(node, n=5):
        neighbors = []
        for nb in G.neighbors(node):
            neighbors.append((nb, G[node][nb]["weight"]))
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return [n[0] for n in neighbors[:n]]

    # Build node data
    nodes_data = []
    for node in G.nodes():
        x, y = pos[node]
        comm = G.nodes[node]["community"]
        freq = freqs[node]
        size = 4 + 18 * (freq / max_freq)
        is_bigram = " " in node
        top5 = get_top_neighbors(node)
        nodes_data.append({
            "id": node,
            "x": float(x),
            "y": float(y),
            "community": comm,
            "theme": theme_labels.get(comm, f"Community {comm}"),
            "color": colors[comm % len(colors)],
            "freq": freq,
            "size": float(size),
            "bigram": is_bigram,
            "top5": top5,
        })

    # Build edges — only keep top edges per node to reduce weight
    edges_data = []
    for u, v in G.edges():
        w = G[u][v]["weight"]
        edges_data.append({
            "source": u,
            "target": v,
            "weight": round(float(w), 3),
        })

    # Sort edges by weight, keep only top 2000 for performance
    edges_data.sort(key=lambda e: e["weight"], reverse=True)
    edges_data = edges_data[:2000]

    graph_json = _json.dumps({"nodes": nodes_data, "edges": edges_data})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HCI Research Landscape — Interactive Map</title>
<style>
* {{ margin: 0; padding: 0; box-sizing: border-box; }}
body {{ background: #0a0a1a; color: #e0e0e0; font-family: 'Inter', -apple-system, sans-serif; overflow: hidden; }}
#canvas {{ display: block; cursor: grab; }}
#canvas.grabbing {{ cursor: grabbing; }}

#search-container {{
    position: fixed; top: 20px; left: 50%; transform: translateX(-50%);
    z-index: 100; display: flex; gap: 8px; align-items: center;
}}
#search-input {{
    width: 360px; padding: 12px 20px; border-radius: 30px;
    border: 1px solid rgba(255,255,255,0.15); background: rgba(10,10,30,0.85);
    backdrop-filter: blur(12px); color: #fff; font-size: 15px;
    outline: none; transition: border-color 0.2s;
}}
#search-input:focus {{ border-color: rgba(255,255,255,0.4); }}
#search-input::placeholder {{ color: rgba(255,255,255,0.35); }}
#search-results {{
    position: fixed; top: 60px; left: 50%; transform: translateX(-50%);
    z-index: 99; background: rgba(10,10,30,0.92); backdrop-filter: blur(12px);
    border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
    max-height: 240px; overflow-y: auto; display: none; min-width: 360px;
}}
.search-item {{
    padding: 10px 20px; cursor: pointer; font-size: 14px;
    border-bottom: 1px solid rgba(255,255,255,0.05); transition: background 0.15s;
}}
.search-item:hover {{ background: rgba(255,255,255,0.08); }}
.search-item .freq {{ color: rgba(255,255,255,0.4); font-size: 12px; margin-left: 8px; }}

#card {{
    position: fixed; right: 24px; top: 80px; width: 320px;
    background: rgba(10,10,30,0.92); backdrop-filter: blur(16px);
    border-radius: 16px; border: 1px solid rgba(255,255,255,0.12);
    padding: 24px; z-index: 100; display: none;
    box-shadow: 0 8px 32px rgba(0,0,0,0.5);
}}
#card h2 {{ font-size: 20px; margin-bottom: 4px; font-weight: 700; }}
#card .theme-badge {{
    display: inline-block; padding: 3px 12px; border-radius: 20px;
    font-size: 12px; font-weight: 600; margin: 8px 0 16px 0;
}}
#card .stat {{ font-size: 13px; color: rgba(255,255,255,0.55); margin-bottom: 12px; }}
#card .section-title {{ font-size: 12px; text-transform: uppercase; letter-spacing: 1px;
    color: rgba(255,255,255,0.4); margin: 16px 0 8px 0; }}
#card .keyword-link {{
    display: flex; align-items: center; padding: 8px 12px;
    background: rgba(255,255,255,0.04); border-radius: 8px; margin-bottom: 6px;
    font-size: 14px; cursor: pointer; transition: background 0.15s;
}}
#card .keyword-link:hover {{ background: rgba(255,255,255,0.1); }}
#card .keyword-link .dot {{ width: 8px; height: 8px; border-radius: 50%;
    margin-right: 10px; flex-shrink: 0; }}
#card .keyword-link .pmi {{ margin-left: auto; color: rgba(255,255,255,0.35); font-size: 12px; }}
#card-close {{
    position: absolute; top: 12px; right: 16px; background: none; border: none;
    color: rgba(255,255,255,0.4); font-size: 20px; cursor: pointer; padding: 4px;
}}
#card-close:hover {{ color: #fff; }}

#legend {{
    position: fixed; bottom: 20px; left: 20px; z-index: 100;
    background: rgba(10,10,30,0.85); backdrop-filter: blur(12px);
    border-radius: 12px; border: 1px solid rgba(255,255,255,0.1);
    padding: 16px; max-height: 300px; overflow-y: auto;
}}
#legend h3 {{ font-size: 13px; margin-bottom: 10px; color: rgba(255,255,255,0.6); text-transform: uppercase; letter-spacing: 1px; }}
.legend-item {{ display: flex; align-items: center; gap: 8px; padding: 3px 0; font-size: 13px; cursor: pointer; }}
.legend-item:hover {{ color: #fff; }}
.legend-dot {{ width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }}

#title {{
    position: fixed; top: 20px; left: 20px; z-index: 100;
    font-size: 13px; color: rgba(255,255,255,0.4);
}}
#zoom-hint {{
    position: fixed; bottom: 20px; right: 20px; z-index: 100;
    font-size: 12px; color: rgba(255,255,255,0.25);
}}
</style>
</head>
<body>

<div id="title">HCI Research Landscape &mdash; 164,859 papers analyzed</div>

<div id="search-container">
    <input id="search-input" type="text" placeholder="Search keywords... (e.g. virtual reality, accessibility)" autocomplete="off">
</div>
<div id="search-results"></div>

<div id="card">
    <button id="card-close">&times;</button>
    <h2 id="card-title"></h2>
    <div class="theme-badge" id="card-theme"></div>
    <div class="stat" id="card-stat"></div>
    <div class="section-title">Top Connected Terms</div>
    <div id="card-keywords"></div>
</div>

<div id="legend"></div>
<div id="zoom-hint">Scroll to zoom &middot; Drag to pan &middot; Click a node to inspect</div>

<canvas id="canvas"></canvas>

<script>
const DATA = {graph_json};

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let W, H, dpr;
function resize() {{
    dpr = window.devicePixelRatio || 1;
    W = window.innerWidth; H = window.innerHeight;
    canvas.width = W * dpr; canvas.height = H * dpr;
    canvas.style.width = W + 'px'; canvas.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}}
resize();
window.addEventListener('resize', () => {{ resize(); draw(); }});

// Transform state
let panX = W / 2, panY = H / 2, scale = 380;
let selectedNode = null;
let highlightedNodes = new Set();
let highlightedEdges = new Set();

// Build lookup
const nodeMap = {{}};
DATA.nodes.forEach(n => {{ nodeMap[n.id] = n; }});
const edgesByNode = {{}};
DATA.edges.forEach(e => {{
    if (!edgesByNode[e.source]) edgesByNode[e.source] = [];
    if (!edgesByNode[e.target]) edgesByNode[e.target] = [];
    edgesByNode[e.source].push(e);
    edgesByNode[e.target].push(e);
}});

// Convert graph coords to screen
function toScreen(x, y) {{ return [x * scale + panX, y * scale + panY]; }}
function fromScreen(sx, sy) {{ return [(sx - panX) / scale, (sy - panY) / scale]; }}

function draw() {{
    ctx.clearRect(0, 0, W, H);

    // Draw edges
    DATA.edges.forEach(e => {{
        const s = nodeMap[e.source], t = nodeMap[e.target];
        if (!s || !t) return;
        const [x1, y1] = toScreen(s.x, s.y);
        const [x2, y2] = toScreen(t.x, t.y);

        const isHighlighted = highlightedEdges.has(e.source + '|' + e.target) ||
                              highlightedEdges.has(e.target + '|' + e.source);

        if (isHighlighted) {{
            ctx.strokeStyle = 'rgba(255,255,255,0.6)';
            ctx.lineWidth = 2;
        }} else if (highlightedNodes.size > 0) {{
            ctx.strokeStyle = 'rgba(255,255,255,0.015)';
            ctx.lineWidth = 0.5;
        }} else {{
            ctx.strokeStyle = 'rgba(255,255,255,0.04)';
            ctx.lineWidth = 0.5;
        }}
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
    }});

    // Draw nodes
    DATA.nodes.forEach(n => {{
        const [x, y] = toScreen(n.x, n.y);
        if (x < -50 || x > W + 50 || y < -50 || y > H + 50) return;

        const isSelected = selectedNode && selectedNode.id === n.id;
        const isHighlighted = highlightedNodes.has(n.id);
        const dimmed = highlightedNodes.size > 0 && !isHighlighted && !isSelected;

        let r = n.size * (scale / 380);
        if (isSelected) r *= 1.6;
        else if (isHighlighted) r *= 1.3;

        let alpha = dimmed ? 0.1 : (isHighlighted || isSelected ? 1.0 : 0.75);

        // Glow for selected/highlighted
        if (isSelected || isHighlighted) {{
            ctx.beginPath();
            ctx.arc(x, y, r + 6, 0, Math.PI * 2);
            ctx.fillStyle = n.color + '30';
            ctx.fill();
        }}

        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = hexWithAlpha(n.color, alpha);
        ctx.fill();

        if (isSelected) {{
            ctx.strokeStyle = '#fff';
            ctx.lineWidth = 2;
            ctx.stroke();
        }}

        // Labels
        const showLabel = isSelected || isHighlighted || (scale > 300 && n.size > 10) || scale > 600;
        if (showLabel && !dimmed) {{
            ctx.font = `${{isSelected ? 'bold 14' : isHighlighted ? 'bold 12' : '11'}}px Inter, sans-serif`;
            ctx.fillStyle = isSelected || isHighlighted ? '#fff' : 'rgba(255,255,255,0.65)';
            ctx.textAlign = 'center';
            ctx.fillText(n.id, x, y - r - 5);
        }}
    }});
}}

function hexWithAlpha(hex, alpha) {{
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
}}

// --- Interaction: Pan & Zoom ---
let dragging = false, lastMx, lastMy;

canvas.addEventListener('mousedown', e => {{
    dragging = true; lastMx = e.clientX; lastMy = e.clientY;
    canvas.classList.add('grabbing');
}});
window.addEventListener('mousemove', e => {{
    if (!dragging) return;
    panX += e.clientX - lastMx;
    panY += e.clientY - lastMy;
    lastMx = e.clientX; lastMy = e.clientY;
    draw();
}});
window.addEventListener('mouseup', () => {{
    dragging = false;
    canvas.classList.remove('grabbing');
}});

canvas.addEventListener('wheel', e => {{
    e.preventDefault();
    const [mx, my] = [e.clientX, e.clientY];
    const [wx, wy] = fromScreen(mx, my);
    const factor = e.deltaY < 0 ? 1.12 : 0.89;
    scale *= factor;
    panX = mx - wx * scale;
    panY = my - wy * scale;
    draw();
}}, {{ passive: false }});

// --- Click: Select Node ---
canvas.addEventListener('click', e => {{
    const [mx, my] = [e.clientX, e.clientY];
    let closest = null, closestDist = Infinity;
    DATA.nodes.forEach(n => {{
        const [x, y] = toScreen(n.x, n.y);
        const d = Math.hypot(mx - x, my - y);
        const r = n.size * (scale / 380) + 6;
        if (d < r && d < closestDist) {{ closest = n; closestDist = d; }}
    }});

    if (closest) {{
        selectNode(closest, true);
    }} else {{
        clearSelection();
    }}
}});

function selectNode(node, animate) {{
    selectedNode = node;
    highlightedNodes = new Set([node.id, ...node.top5]);
    highlightedEdges = new Set();
    // highlight edges between selected and top5
    node.top5.forEach(nb => {{
        highlightedEdges.add(node.id + '|' + nb);
    }});

    if (animate) {{
        // Smooth zoom to node
        const targetScale = Math.max(scale, 500);
        const [tx, ty] = [W/2 - node.x * targetScale, H/2 - node.y * targetScale];
        animateTo(tx, ty, targetScale);
    }}

    showCard(node);
    draw();
}}

function clearSelection() {{
    selectedNode = null;
    highlightedNodes = new Set();
    highlightedEdges = new Set();
    document.getElementById('card').style.display = 'none';
    draw();
}}

function animateTo(tx, ty, ts) {{
    const startPx = panX, startPy = panY, startS = scale;
    const dur = 400;
    const start = performance.now();
    function step(t) {{
        const p = Math.min((t - start) / dur, 1);
        const ease = 1 - Math.pow(1 - p, 3);
        panX = startPx + (tx - startPx) * ease;
        panY = startPy + (ty - startPy) * ease;
        scale = startS + (ts - startS) * ease;
        draw();
        if (p < 1) requestAnimationFrame(step);
    }}
    requestAnimationFrame(step);
}}

// --- Card ---
function showCard(node) {{
    const card = document.getElementById('card');
    document.getElementById('card-title').textContent = node.id;

    const badge = document.getElementById('card-theme');
    badge.textContent = node.theme;
    badge.style.background = node.color + '30';
    badge.style.color = node.color;

    document.getElementById('card-stat').textContent =
        `Frequency: ${{node.freq.toLocaleString()}} papers  \\u00B7  Community ${{node.community}}`;

    const kwDiv = document.getElementById('card-keywords');
    kwDiv.innerHTML = '';
    if (node.top5.length === 0) {{
        kwDiv.innerHTML = '<div style="color:rgba(255,255,255,0.3);font-size:13px;padding:8px">No bigram neighbors</div>';
    }}
    node.top5.forEach(kw => {{
        const nb = nodeMap[kw];
        if (!nb) return;
        // find edge weight
        let w = 0;
        const edges = edgesByNode[node.id] || [];
        edges.forEach(e => {{
            if (e.source === kw || e.target === kw) w = e.weight;
        }});
        const el = document.createElement('div');
        el.className = 'keyword-link';
        el.innerHTML = `<span class="dot" style="background:${{nb.color}}"></span>${{kw}}<span class="pmi">PMI ${{w.toFixed(2)}}</span>`;
        el.onclick = () => {{
            selectNode(nb, true);
            searchInput.value = kw;
        }};
        kwDiv.appendChild(el);
    }});

    card.style.display = 'block';
}}

document.getElementById('card-close').onclick = clearSelection;

// --- Search ---
const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');

searchInput.addEventListener('input', () => {{
    const q = searchInput.value.toLowerCase().trim();
    if (q.length < 2) {{ searchResults.style.display = 'none'; return; }}

    const matches = DATA.nodes
        .filter(n => n.id.includes(q))
        .sort((a, b) => {{
            // exact match first, then startsWith, then by frequency
            const aExact = a.id === q ? 2 : a.id.startsWith(q) ? 1 : 0;
            const bExact = b.id === q ? 2 : b.id.startsWith(q) ? 1 : 0;
            if (aExact !== bExact) return bExact - aExact;
            return b.freq - a.freq;
        }})
        .slice(0, 8);

    if (matches.length === 0) {{
        searchResults.style.display = 'none'; return;
    }}

    searchResults.innerHTML = '';
    matches.forEach(n => {{
        const el = document.createElement('div');
        el.className = 'search-item';
        el.innerHTML = `<span style="color:${{n.color}}">&#9679;</span> ${{n.id}} <span class="freq">${{n.freq.toLocaleString()}} papers</span>`;
        el.onclick = () => {{
            searchInput.value = n.id;
            searchResults.style.display = 'none';
            selectNode(n, true);
        }};
        searchResults.appendChild(el);
    }});
    searchResults.style.display = 'block';
}});

searchInput.addEventListener('keydown', e => {{
    if (e.key === 'Enter') {{
        const q = searchInput.value.toLowerCase().trim();
        const match = DATA.nodes.find(n => n.id === q) ||
                      DATA.nodes.find(n => n.id.includes(q));
        if (match) {{
            searchResults.style.display = 'none';
            selectNode(match, true);
        }}
    }}
    if (e.key === 'Escape') {{
        searchResults.style.display = 'none';
        clearSelection();
        searchInput.value = '';
        searchInput.blur();
    }}
}});

document.addEventListener('click', e => {{
    if (!e.target.closest('#search-container') && !e.target.closest('#search-results'))
        searchResults.style.display = 'none';
}});

// --- Legend ---
function buildLegend() {{
    const legend = document.getElementById('legend');
    const themes = {{}};
    DATA.nodes.forEach(n => {{
        if (!themes[n.community]) themes[n.community] = {{ color: n.color, theme: n.theme, count: 0 }};
        themes[n.community].count++;
    }});
    let html = '<h3>Themes</h3>';
    Object.entries(themes).sort((a,b) => b[1].count - a[1].count).forEach(([id, t]) => {{
        html += `<div class="legend-item" onclick="highlightCommunity(${{id}})">
            <span class="legend-dot" style="background:${{t.color}}"></span>
            ${{t.theme}} <span style="color:rgba(255,255,255,0.3);margin-left:auto;font-size:11px">${{t.count}}</span>
        </div>`;
    }});
    legend.innerHTML = html;
}}

function highlightCommunity(commId) {{
    highlightedNodes = new Set();
    highlightedEdges = new Set();
    selectedNode = null;
    DATA.nodes.forEach(n => {{
        if (n.community === commId) highlightedNodes.add(n.id);
    }});
    document.getElementById('card').style.display = 'none';
    draw();
    // Double-click to clear
    setTimeout(() => {{
        // auto-clear after 5s if no interaction
    }}, 5000);
}}

buildLegend();
draw();
</script>
</body>
</html>"""

    output_path = os.path.join(OUTPUT_DIR, "map2_interactive.html")
    with open(output_path, "w") as f:
        f.write(html)
    file_size = os.path.getsize(output_path) / 1024
    print(f"Saved: map2_interactive.html ({file_size:.0f} KB)")


def save_community_report(G, communities):
    """Save a text report of communities."""
    report_lines = ["HCI Research Communities\n", "=" * 50 + "\n\n"]

    for i, comm in enumerate(communities):
        terms_sorted = sorted(comm, key=lambda n: G.nodes[n].get("freq", 0), reverse=True)
        report_lines.append(f"Community {i} — {len(comm)} terms\n")
        report_lines.append("-" * 40 + "\n")
        for t in terms_sorted[:30]:
            report_lines.append(f"  {t} (freq: {G.nodes[t].get('freq', 0)})\n")
        report_lines.append("\n")

    report_path = os.path.join(OUTPUT_DIR, "communities_report.txt")
    with open(report_path, "w") as f:
        f.writelines(report_lines)
    print(f"Saved: communities_report.txt")


def main():
    dtm, vocab = load_data()
    G = build_network(dtm, vocab, top_n=400, min_cooc=100)
    communities = detect_communities(G)
    plot_static_network(G, communities)
    create_interactive_network(G, communities)
    save_community_report(G, communities)
    print("\nDone! Check the output/ directory for results.")


if __name__ == "__main__":
    main()

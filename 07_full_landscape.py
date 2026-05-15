"""
Step 7: Full 30M-paper landscape — saturated interactive network map.

Samples from the ENTIRE unfiltered dataset (30M papers) to show the full
breadth of topics, including those adjacent to HCI from other fields.
Generates a standalone interactive HTML.
"""

import json
import os
import random
import re

import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAIN_FILE = os.path.join(DATA_DIR, "hci_papers.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

SAMPLE_SIZE = 2_000_000

COLORS = [
    "#FF1744", "#00E676", "#FFEA00", "#2979FF", "#FF9100",
    "#D500F9", "#00E5FF", "#F50057", "#76FF03", "#FF6D00",
    "#651FFF", "#1DE9B6", "#C6FF00", "#304FFE", "#DD2C00",
    "#00BFA5", "#FFD600", "#AA00FF", "#64DD17", "#6200EA",
]


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


from stopwords import is_valid_term


def load_sample(path, n):
    """Reservoir-sample n papers from the full file."""
    print("Counting papers...")
    total = 0
    with open(path, "r") as f:
        for _ in f:
            total += 1
    print(f"Total: {total:,}")

    rate = n / total
    random.seed(42)
    papers = []
    print(f"Sampling ~{n:,} papers...")
    with open(path, "r") as f:
        for line in tqdm(f, total=total, desc="Sampling"):
            if random.random() > rate:
                continue
            p = json.loads(line.strip())
            if p.get("abstract") and len(p["abstract"]) > 50:
                papers.append(clean_text(p["abstract"]))
            if len(papers) >= n:
                break
    print(f"Sampled {len(papers):,} papers")
    return papers


def build_dtm(abstracts):
    print("Building DTM...")
    vec = TfidfVectorizer(
        min_df=10, max_df=0.5, max_features=20000,
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
    )
    dtm = vec.fit_transform(abstracts)
    vocab = np.array(vec.get_feature_names_out())
    print(f"DTM: {dtm.shape[0]:,} x {dtm.shape[1]:,}")
    return dtm, vocab


def build_network(dtm, vocab, top_n=500, min_cooc=100):
    print(f"\nBuilding network (top {top_n} terms, min_cooc={min_cooc})...")
    term_freq = np.asarray((dtm > 0).sum(axis=0)).flatten()

    valid_mask = np.array([is_valid_term(v) for v in vocab])
    filtered_freq = term_freq.copy()
    filtered_freq[~valid_mask] = 0

    top_indices = filtered_freq.argsort()[-top_n:][::-1]
    top_indices = top_indices[filtered_freq[top_indices] > 0]

    dtm_sub = dtm[:, top_indices]
    top_vocab = vocab[top_indices]

    binary = (dtm_sub > 0).astype(int)
    n_docs = binary.shape[0]
    cooc = (binary.T @ binary).toarray().astype(float)
    np.fill_diagonal(cooc, 0)
    doc_freq = np.asarray(binary.sum(axis=0)).flatten().astype(float)

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
            if pmi > 0.5:
                G.add_edge(top_vocab[i], top_vocab[j], weight=float(pmi))

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)

    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    communities = nx.community.louvain_communities(G, seed=42, resolution=1.0)
    for i, comm in enumerate(communities):
        for node in comm:
            G.nodes[node]["community"] = i

    print(f"Detected {len(communities)} communities")
    for i, comm in enumerate(communities):
        top = sorted(comm, key=lambda n: G.nodes[n].get("freq", 0), reverse=True)[:5]
        print(f"  C{i} ({len(comm)}): {', '.join(top)}")

    return G, communities


def create_html(G, communities, n_papers):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pos = nx.spring_layout(G, k=1.0, iterations=200, seed=42, weight="weight")

    theme_labels = {}
    for i, comm in enumerate(communities):
        bigrams = [n for n in comm if " " in n]
        if bigrams:
            top = sorted(bigrams, key=lambda n: G.nodes[n].get("freq", 0), reverse=True)[:2]
            theme_labels[i] = " / ".join(top).title()
        else:
            top = sorted(comm, key=lambda n: G.nodes[n].get("freq", 0), reverse=True)[:2]
            theme_labels[i] = " / ".join(top).title()

    freqs = {n: G.nodes[n].get("freq", 1) for n in G.nodes()}
    max_freq = max(freqs.values())

    def get_top_neighbors(node, n=5):
        neighbors = []
        for nb in G.neighbors(node):
            neighbors.append((nb, G[node][nb]["weight"]))
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return [n[0] for n in neighbors[:n]]

    nodes_data = []
    for node in G.nodes():
        x, y = pos[node]
        comm = G.nodes[node]["community"]
        freq = freqs[node]
        size = 4 + 18 * (freq / max_freq)
        top5 = get_top_neighbors(node)
        nodes_data.append({
            "id": node, "x": float(x), "y": float(y),
            "community": comm,
            "theme": theme_labels.get(comm, f"Community {comm}"),
            "color": COLORS[comm % len(COLORS)],
            "freq": freq, "size": float(size),
            "bigram": " " in node, "top5": top5,
        })

    edges_data = []
    for u, v in G.edges():
        w = G[u][v]["weight"]
        edges_data.append({"source": u, "target": v, "weight": round(float(w), 3)})
    edges_data.sort(key=lambda e: e["weight"], reverse=True)
    edges_data = edges_data[:2500]

    import json as _json
    graph_json = _json.dumps({"nodes": nodes_data, "edges": edges_data})

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HCI Research Landscape — Full 30M Paper View</title>
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
    width: 400px; padding: 12px 20px; border-radius: 30px;
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
    max-height: 240px; overflow-y: auto; display: none; min-width: 400px;
}}
.search-item {{
    padding: 10px 20px; cursor: pointer; font-size: 14px;
    border-bottom: 1px solid rgba(255,255,255,0.05); transition: background 0.15s;
}}
.search-item:hover {{ background: rgba(255,255,255,0.08); }}
.search-item .freq {{ color: rgba(255,255,255,0.4); font-size: 12px; margin-left: 8px; }}

#card {{
    position: fixed; right: 24px; top: 80px; width: 340px;
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
    padding: 16px; max-height: 350px; overflow-y: auto;
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

<div id="title">HCI + Adjacent Fields &mdash; 30M papers, {G.number_of_nodes()} terms, {len(communities)} clusters</div>

<div id="search-container">
    <input id="search-input" type="text" placeholder="Search terms... (e.g. virtual reality, machine learning, accessibility)" autocomplete="off">
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

let panX = W / 2, panY = H / 2, scale = 380;
let selectedNode = null;
let highlightedNodes = new Set();
let highlightedEdges = new Set();

const nodeMap = {{}};
DATA.nodes.forEach(n => {{ nodeMap[n.id] = n; }});
const edgesByNode = {{}};
DATA.edges.forEach(e => {{
    if (!edgesByNode[e.source]) edgesByNode[e.source] = [];
    if (!edgesByNode[e.target]) edgesByNode[e.target] = [];
    edgesByNode[e.source].push(e);
    edgesByNode[e.target].push(e);
}});

function toScreen(x, y) {{ return [x * scale + panX, y * scale + panY]; }}
function fromScreen(sx, sy) {{ return [(sx - panX) / scale, (sy - panY) / scale]; }}

function hexWithAlpha(hex, alpha) {{
    const r = parseInt(hex.slice(1,3), 16);
    const g = parseInt(hex.slice(3,5), 16);
    const b = parseInt(hex.slice(5,7), 16);
    return `rgba(${{r}},${{g}},${{b}},${{alpha}})`;
}}

function draw() {{
    ctx.clearRect(0, 0, W, H);
    DATA.edges.forEach(e => {{
        const s = nodeMap[e.source], t = nodeMap[e.target];
        if (!s || !t) return;
        const [x1, y1] = toScreen(s.x, s.y);
        const [x2, y2] = toScreen(t.x, t.y);
        const isHL = highlightedEdges.has(e.source + '|' + e.target) || highlightedEdges.has(e.target + '|' + e.source);
        if (isHL) {{ ctx.strokeStyle = 'rgba(255,255,255,0.6)'; ctx.lineWidth = 2; }}
        else if (highlightedNodes.size > 0) {{ ctx.strokeStyle = 'rgba(255,255,255,0.015)'; ctx.lineWidth = 0.5; }}
        else {{ ctx.strokeStyle = 'rgba(255,255,255,0.04)'; ctx.lineWidth = 0.5; }}
        ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
    }});
    DATA.nodes.forEach(n => {{
        const [x, y] = toScreen(n.x, n.y);
        if (x < -50 || x > W + 50 || y < -50 || y > H + 50) return;
        const isSel = selectedNode && selectedNode.id === n.id;
        const isHL = highlightedNodes.has(n.id);
        const dimmed = highlightedNodes.size > 0 && !isHL && !isSel;
        let r = n.size * (scale / 380);
        if (isSel) r *= 1.6; else if (isHL) r *= 1.3;
        let alpha = dimmed ? 0.1 : (isHL || isSel ? 1.0 : 0.75);
        if (isSel || isHL) {{
            ctx.beginPath(); ctx.arc(x, y, r + 6, 0, Math.PI * 2);
            ctx.fillStyle = n.color + '30'; ctx.fill();
        }}
        ctx.beginPath(); ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = hexWithAlpha(n.color, alpha); ctx.fill();
        if (isSel) {{ ctx.strokeStyle = '#fff'; ctx.lineWidth = 2; ctx.stroke(); }}
        const showLabel = isSel || isHL || (scale > 300 && n.size > 10) || scale > 600;
        if (showLabel && !dimmed) {{
            ctx.font = `${{isSel ? 'bold 14' : isHL ? 'bold 12' : '11'}}px Inter, sans-serif`;
            ctx.fillStyle = isSel || isHL ? '#fff' : 'rgba(255,255,255,0.65)';
            ctx.textAlign = 'center';
            ctx.fillText(n.id, x, y - r - 5);
        }}
    }});
}}

let dragging = false, lastMx, lastMy;
canvas.addEventListener('mousedown', e => {{ dragging = true; lastMx = e.clientX; lastMy = e.clientY; canvas.classList.add('grabbing'); }});
window.addEventListener('mousemove', e => {{ if (!dragging) return; panX += e.clientX - lastMx; panY += e.clientY - lastMy; lastMx = e.clientX; lastMy = e.clientY; draw(); }});
window.addEventListener('mouseup', () => {{ dragging = false; canvas.classList.remove('grabbing'); }});
canvas.addEventListener('wheel', e => {{
    e.preventDefault();
    const [mx, my] = [e.clientX, e.clientY];
    const [wx, wy] = fromScreen(mx, my);
    const factor = e.deltaY < 0 ? 1.12 : 0.89;
    scale *= factor;
    panX = mx - wx * scale; panY = my - wy * scale;
    draw();
}}, {{ passive: false }});

canvas.addEventListener('click', e => {{
    const [mx, my] = [e.clientX, e.clientY];
    let closest = null, closestDist = Infinity;
    DATA.nodes.forEach(n => {{
        const [x, y] = toScreen(n.x, n.y);
        const d = Math.hypot(mx - x, my - y);
        const r = n.size * (scale / 380) + 6;
        if (d < r && d < closestDist) {{ closest = n; closestDist = d; }}
    }});
    if (closest) selectNode(closest, true); else clearSelection();
}});

function selectNode(node, animate) {{
    selectedNode = node;
    highlightedNodes = new Set([node.id, ...node.top5]);
    highlightedEdges = new Set();
    node.top5.forEach(nb => {{ highlightedEdges.add(node.id + '|' + nb); }});
    if (animate) {{
        const ts = Math.max(scale, 500);
        animateTo(W/2 - node.x * ts, H/2 - node.y * ts, ts);
    }}
    showCard(node); draw();
}}
function clearSelection() {{
    selectedNode = null; highlightedNodes = new Set(); highlightedEdges = new Set();
    document.getElementById('card').style.display = 'none'; draw();
}}
function animateTo(tx, ty, ts) {{
    const sx = panX, sy = panY, ss = scale, dur = 400, t0 = performance.now();
    function step(t) {{
        const p = Math.min((t - t0) / dur, 1), ease = 1 - Math.pow(1 - p, 3);
        panX = sx + (tx - sx) * ease; panY = sy + (ty - sy) * ease;
        scale = ss + (ts - ss) * ease; draw();
        if (p < 1) requestAnimationFrame(step);
    }}
    requestAnimationFrame(step);
}}

function showCard(node) {{
    const card = document.getElementById('card');
    document.getElementById('card-title').textContent = node.id;
    const badge = document.getElementById('card-theme');
    badge.textContent = node.theme; badge.style.background = node.color + '30'; badge.style.color = node.color;
    document.getElementById('card-stat').textContent = `Frequency: ${{node.freq.toLocaleString()}} papers  \\u00B7  Community ${{node.community}}`;
    const kwDiv = document.getElementById('card-keywords');
    kwDiv.innerHTML = '';
    if (node.top5.length === 0) {{ kwDiv.innerHTML = '<div style="color:rgba(255,255,255,0.3);font-size:13px;padding:8px">No neighbors</div>'; }}
    node.top5.forEach(kw => {{
        const nb = nodeMap[kw]; if (!nb) return;
        let w = 0;
        (edgesByNode[node.id] || []).forEach(e => {{ if (e.source === kw || e.target === kw) w = e.weight; }});
        const el = document.createElement('div');
        el.className = 'keyword-link';
        el.innerHTML = `<span class="dot" style="background:${{nb.color}}"></span>${{kw}}<span class="pmi">PMI ${{w.toFixed(2)}}</span>`;
        el.onclick = () => {{ selectNode(nb, true); searchInput.value = kw; }};
        kwDiv.appendChild(el);
    }});
    card.style.display = 'block';
}}
document.getElementById('card-close').onclick = clearSelection;

const searchInput = document.getElementById('search-input');
const searchResults = document.getElementById('search-results');
searchInput.addEventListener('input', () => {{
    const q = searchInput.value.toLowerCase().trim();
    if (q.length < 2) {{ searchResults.style.display = 'none'; return; }}
    const matches = DATA.nodes.filter(n => n.id.includes(q))
        .sort((a, b) => {{ const ae = a.id === q ? 2 : a.id.startsWith(q) ? 1 : 0; const be = b.id === q ? 2 : b.id.startsWith(q) ? 1 : 0; if (ae !== be) return be - ae; return b.freq - a.freq; }})
        .slice(0, 8);
    if (matches.length === 0) {{ searchResults.style.display = 'none'; return; }}
    searchResults.innerHTML = '';
    matches.forEach(n => {{
        const el = document.createElement('div'); el.className = 'search-item';
        el.innerHTML = `<span style="color:${{n.color}}">&#9679;</span> ${{n.id}} <span class="freq">${{n.freq.toLocaleString()}} papers</span>`;
        el.onclick = () => {{ searchInput.value = n.id; searchResults.style.display = 'none'; selectNode(n, true); }};
        searchResults.appendChild(el);
    }});
    searchResults.style.display = 'block';
}});
searchInput.addEventListener('keydown', e => {{
    if (e.key === 'Enter') {{ const q = searchInput.value.toLowerCase().trim(); const m = DATA.nodes.find(n => n.id === q) || DATA.nodes.find(n => n.id.includes(q)); if (m) {{ searchResults.style.display = 'none'; selectNode(m, true); }} }}
    if (e.key === 'Escape') {{ searchResults.style.display = 'none'; clearSelection(); searchInput.value = ''; searchInput.blur(); }}
}});
document.addEventListener('click', e => {{ if (!e.target.closest('#search-container') && !e.target.closest('#search-results')) searchResults.style.display = 'none'; }});

function buildLegend() {{
    const legend = document.getElementById('legend');
    const themes = {{}};
    DATA.nodes.forEach(n => {{ if (!themes[n.community]) themes[n.community] = {{ color: n.color, theme: n.theme, count: 0 }}; themes[n.community].count++; }});
    let html = '<h3>Clusters</h3>';
    Object.entries(themes).sort((a,b) => b[1].count - a[1].count).forEach(([id, t]) => {{
        html += `<div class="legend-item" onclick="highlightCommunity(${{id}})"><span class="legend-dot" style="background:${{t.color}}"></span>${{t.theme}} <span style="color:rgba(255,255,255,0.3);margin-left:auto;font-size:11px">${{t.count}}</span></div>`;
    }});
    legend.innerHTML = html;
}}
function highlightCommunity(commId) {{
    highlightedNodes = new Set(); highlightedEdges = new Set(); selectedNode = null;
    DATA.nodes.forEach(n => {{ if (n.community === commId) highlightedNodes.add(n.id); }});
    document.getElementById('card').style.display = 'none'; draw();
}}
buildLegend(); draw();
</script>
</body>
</html>"""

    output_path = os.path.join(OUTPUT_DIR, "map_full_30m.html")
    with open(output_path, "w") as f:
        f.write(html)
    sz = os.path.getsize(output_path) / 1024
    print(f"\nSaved: {output_path} ({sz:.0f} KB)")


def main():
    abstracts = load_sample(MAIN_FILE, SAMPLE_SIZE)
    dtm, vocab = build_dtm(abstracts)
    G, communities = build_network(dtm, vocab, top_n=500, min_cooc=100)
    create_html(G, communities, len(abstracts))
    print("\nDone! Open output/map_full_30m.html in a browser.")


if __name__ == "__main__":
    main()

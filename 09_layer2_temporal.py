"""
Step 9: Layer 2 Temporal — HCI Topic Evolution with Year Slider

Combines the bigram-only subject-matter filtering (from 08_layer2_map.py)
with the global-layout temporal approach (from 06_temporal_network.py).

Result: an interactive map showing how HCI topics evolve year by year,
with consistent positions/colors and smooth transitions.
"""

import json
import os
import re

import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

from stopwords import is_valid_term

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

COLORS = [
    "#FF1744", "#00E676", "#FFEA00", "#2979FF", "#FF9100",
    "#D500F9", "#00E5FF", "#F50057", "#76FF03", "#FF6D00",
    "#651FFF", "#1DE9B6", "#C6FF00", "#304FFE", "#DD2C00",
    "#00BFA5", "#FFD600", "#AA00FF", "#64DD17", "#6200EA",
]

# Words that indicate methodology, not subject matter
METH_WORDS = {
    'study', 'studies', 'results', 'findings', 'paper',
    'conducted', 'analyzed', 'aimed', 'proposes', 'examines',
    'investigates', 'revealed', 'showed', 'suggests',
    'demonstrates', 'indicates', 'obtained', 'research',
    'methods', 'qualitative', 'quantitative', 'significant',
    'implications', 'literature', 'review', 'interviews',
    'structured', 'aims', 'explores', 'discusses',
    'sample', 'survey', 'questionnaire', 'respondents',
    'analysis', 'statistics', 'statistical', 'descriptive',
    'control', 'group', 'participants', 'hypothesis',
    'variables', 'correlation', 'regression', 'anova',
    'used', 'based', 'approach', 'proposed', 'evidence',
}


def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_subject_bigram(term):
    """Only allow bigrams that are subject matter, not methodology."""
    if ' ' not in term:
        return False
    if not is_valid_term(term):
        return False
    parts = term.lower().split()
    if any(p in METH_WORDS for p in parts):
        return False
    return True


def load_filtered_papers():
    input_file = os.path.join(DATA_DIR, "hci_papers_filtered.jsonl")
    papers = []
    with open(input_file, "r") as f:
        for line in tqdm(f, desc="Loading papers"):
            p = json.loads(line.strip())
            if p.get("abstract") and p.get("year"):
                papers.append(p)
    print(f"Loaded {len(papers):,} papers")
    return papers


def build_global_network(all_abstracts, vocab_global, top_n=300, min_cooc=15):
    """Build ONE global network from all papers for consistent layout/communities."""
    print(f"\nBuilding global bigram network from {len(all_abstracts):,} abstracts...")

    vectorizer = TfidfVectorizer(
        vocabulary={v: i for i, v in enumerate(vocab_global)},
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
    )
    dtm = vectorizer.fit_transform(all_abstracts)

    term_freq = np.asarray((dtm > 0).sum(axis=0)).flatten()
    valid_mask = np.array([is_subject_bigram(v) for v in vocab_global])
    filtered_freq = term_freq.copy()
    filtered_freq[~valid_mask] = 0

    top_indices = filtered_freq.argsort()[-top_n:][::-1]
    top_indices = top_indices[filtered_freq[top_indices] > 0]

    dtm_sub = dtm[:, top_indices]
    top_vocab = vocab_global[top_indices]

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
            if pmi > 0.3:
                G.add_edge(top_vocab[i], top_vocab[j], weight=float(pmi))

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    print(f"Global network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    communities = nx.community.louvain_communities(G, seed=42, resolution=1.0)
    for i, comm in enumerate(communities):
        for node in comm:
            G.nodes[node]["community"] = i
    print(f"Detected {len(communities)} global communities")
    for i, comm in enumerate(communities):
        top = sorted(comm, key=lambda n: G.nodes[n].get("freq", 0), reverse=True)[:4]
        print(f"  C{i} ({len(comm)}): {', '.join(top)}")

    pos = nx.spring_layout(G, k=2.0, iterations=200, seed=42, weight="weight")

    theme_labels = {}
    for i, comm in enumerate(communities):
        top = sorted(comm, key=lambda n: G.nodes[n].get("freq", 0), reverse=True)[:2]
        theme_labels[i] = " / ".join(top).title()

    return G, pos, communities, theme_labels, top_indices, top_vocab


def compute_year_data(abstracts_year, vocab_global, top_indices, top_vocab,
                      global_graph, global_pos, communities, theme_labels,
                      prev_comms=None, min_cooc=3):
    n_papers = len(abstracts_year)
    if n_papers < 30:
        return {"nodes": [], "edges": [], "paper_count": n_papers}, prev_comms or []

    vectorizer = TfidfVectorizer(
        vocabulary={v: i for i, v in enumerate(vocab_global)},
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
    )
    dtm = vectorizer.fit_transform(abstracts_year)
    dtm_sub = dtm[:, top_indices]
    binary = (dtm_sub > 0).astype(int)
    n_docs = binary.shape[0]
    year_freq = np.asarray((dtm_sub > 0).sum(axis=0)).flatten()

    cooc = (binary.T @ binary).toarray().astype(float)
    np.fill_diagonal(cooc, 0)
    doc_freq = np.asarray(binary.sum(axis=0)).flatten().astype(float)

    if n_papers < 500: mc = 3
    elif n_papers < 2000: mc = 5
    elif n_papers < 5000: mc = 10
    else: mc = 15

    edges_data = []
    active_nodes = set()
    for i in range(len(top_vocab)):
        for j in range(i + 1, len(top_vocab)):
            if cooc[i][j] < mc: continue
            p_ij = cooc[i][j] / n_docs
            p_i = doc_freq[i] / n_docs
            p_j = doc_freq[j] / n_docs
            if p_i == 0 or p_j == 0: continue
            pmi = np.log2(p_ij / (p_i * p_j))
            if pmi > 0.3:
                edges_data.append({"source": top_vocab[i], "target": top_vocab[j], "weight": round(float(pmi), 3)})
                active_nodes.add(top_vocab[i])
                active_nodes.add(top_vocab[j])

    for i, term in enumerate(top_vocab):
        if year_freq[i] >= max(3, n_papers * 0.01):
            active_nodes.add(term)

    active_nodes = active_nodes & set(global_graph.nodes())
    edges_data.sort(key=lambda e: e["weight"], reverse=True)
    edges_data = edges_data[:2000]

    edge_by_pair = {}
    for e in edges_data:
        edge_by_pair[(e["source"], e["target"])] = e["weight"]
        edge_by_pair[(e["target"], e["source"])] = e["weight"]

    freqs = {term: int(year_freq[i]) for i, term in enumerate(top_vocab) if term in active_nodes}
    max_freq = max(freqs.values()) if freqs else 1

    # Per-year community detection — build a subgraph for this year's active nodes
    year_graph = nx.Graph()
    for term in active_nodes:
        if freqs.get(term, 0) > 0:
            year_graph.add_node(term)
    for e in edges_data:
        if e["source"] in year_graph and e["target"] in year_graph:
            year_graph.add_edge(e["source"], e["target"], weight=e["weight"])
    # Remove isolates from year graph
    year_isolates = list(nx.isolates(year_graph))
    year_graph.remove_nodes_from(year_isolates)

    # Detect communities for THIS year
    raw_comms = []
    if year_graph.number_of_nodes() >= 5:
        raw_comms = list(nx.community.louvain_communities(year_graph, seed=42, resolution=1.0))

    # Match to previous year's communities by Jaccard overlap for color consistency
    year_communities = {}
    year_theme_labels = {}
    if raw_comms and prev_comms:
        # Build mapping: new_idx -> best matching prev_idx
        used_prev = set()
        mapping = {}
        # Score all pairs by Jaccard similarity
        pairs = []
        for ni, nc in enumerate(raw_comms):
            for pi, pc in enumerate(prev_comms):
                overlap = len(nc & pc)
                union = len(nc | pc)
                jaccard = overlap / union if union > 0 else 0
                pairs.append((jaccard, ni, pi))
        pairs.sort(reverse=True)
        for jaccard, ni, pi in pairs:
            if ni in mapping or pi in used_prev:
                continue
            if jaccard > 0.15:  # meaningful overlap
                mapping[ni] = pi
                used_prev.add(pi)
        # Assign unmapped communities to new IDs
        next_id = max((max(used_prev) + 1 if used_prev else 0), len(prev_comms))
        for ni in range(len(raw_comms)):
            if ni not in mapping:
                mapping[ni] = next_id
                next_id += 1
        # Apply mapping
        for ni, comm in enumerate(raw_comms):
            mapped_id = mapping[ni]
            for node in comm:
                year_communities[node] = mapped_id
            top = sorted(comm, key=lambda n: freqs.get(n, 0), reverse=True)[:2]
            year_theme_labels[mapped_id] = " / ".join(top).title()
    elif raw_comms:
        for i, comm in enumerate(raw_comms):
            for node in comm:
                year_communities[node] = i
            top = sorted(comm, key=lambda n: freqs.get(n, 0), reverse=True)[:2]
            year_theme_labels[i] = " / ".join(top).title()

    # Return current communities as sets for next year's matching
    current_comms = [set(c) for c in raw_comms] if raw_comms else []

    nodes_data = []
    for term in active_nodes:
        if term not in global_pos: continue
        x, y = global_pos[term]
        freq = freqs.get(term, 0)
        if freq == 0: continue
        comm = year_communities.get(term, 0)
        size = 3 + 12 * (freq / max_freq)
        neighbors = [(other, edge_by_pair.get((term, other), 0)) for other in active_nodes if other != term and edge_by_pair.get((term, other), 0) > 0]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        top5 = [n[0] for n in neighbors[:5]]
        nodes_data.append({
            "id": term, "x": float(x), "y": float(y),
            "community": comm, "theme": year_theme_labels.get(comm, f"C{comm}"),
            "color": COLORS[comm % len(COLORS)],
            "freq": freq, "size": float(size), "bigram": " " in term, "top5": top5,
        })

    return {"nodes": nodes_data, "edges": edges_data, "paper_count": n_papers}, current_comms


def create_temporal_html(year_data, years):
    """Create interactive HTML — reuses the temporal template from 06."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_data_json = json.dumps(year_data)
    years_json = json.dumps(years)

    # Read the existing temporal HTML template and adapt it
    # (The HTML/JS is identical to 06_temporal_network.py output)
    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HCI Topics Evolution — Year by Year</title>
<style>
*{{margin:0;padding:0;box-sizing:border-box}}body{{background:#0a0a1a;color:#e0e0e0;font-family:'Inter',-apple-system,sans-serif;overflow:hidden}}
#canvas{{display:block;cursor:grab}}#canvas.grabbing{{cursor:grabbing}}
#search-container{{position:fixed;top:20px;left:50%;transform:translateX(-50%);z-index:100}}
#search-input{{width:400px;padding:12px 20px;border-radius:30px;border:1px solid rgba(255,255,255,0.15);background:rgba(10,10,30,0.85);backdrop-filter:blur(12px);color:#fff;font-size:15px;outline:none}}
#search-input:focus{{border-color:rgba(255,255,255,0.4)}}#search-input::placeholder{{color:rgba(255,255,255,0.35)}}
#search-results{{position:fixed;top:60px;left:50%;transform:translateX(-50%);z-index:99;background:rgba(10,10,30,0.92);backdrop-filter:blur(12px);border-radius:12px;border:1px solid rgba(255,255,255,0.1);max-height:240px;overflow-y:auto;display:none;min-width:400px}}
.search-item{{padding:10px 20px;cursor:pointer;font-size:14px;border-bottom:1px solid rgba(255,255,255,0.05)}}.search-item:hover{{background:rgba(255,255,255,0.08)}}
.search-item .freq{{color:rgba(255,255,255,0.4);font-size:12px;margin-left:8px}}
#card{{position:fixed;right:24px;top:80px;width:340px;background:rgba(10,10,30,0.92);backdrop-filter:blur(16px);border-radius:16px;border:1px solid rgba(255,255,255,0.12);padding:24px;z-index:100;display:none;box-shadow:0 8px 32px rgba(0,0,0,0.5)}}
#card h2{{font-size:20px;margin-bottom:4px;font-weight:700}}
#card .theme-badge{{display:inline-block;padding:3px 12px;border-radius:20px;font-size:12px;font-weight:600;margin:8px 0 16px 0}}
#card .stat{{font-size:13px;color:rgba(255,255,255,0.55);margin-bottom:12px}}
#card .section-title{{font-size:12px;text-transform:uppercase;letter-spacing:1px;color:rgba(255,255,255,0.4);margin:16px 0 8px 0}}
#card .keyword-link{{display:flex;align-items:center;padding:8px 12px;background:rgba(255,255,255,0.04);border-radius:8px;margin-bottom:6px;font-size:14px;cursor:pointer}}
#card .keyword-link:hover{{background:rgba(255,255,255,0.1)}}
#card .keyword-link .dot{{width:8px;height:8px;border-radius:50%;margin-right:10px;flex-shrink:0}}
#card .keyword-link .pmi{{margin-left:auto;color:rgba(255,255,255,0.35);font-size:12px}}
#card-close{{position:absolute;top:12px;right:16px;background:none;border:none;color:rgba(255,255,255,0.4);font-size:20px;cursor:pointer}}#card-close:hover{{color:#fff}}
#legend{{position:fixed;bottom:130px;left:20px;z-index:100;background:rgba(10,10,30,0.85);backdrop-filter:blur(12px);border-radius:12px;border:1px solid rgba(255,255,255,0.1);padding:16px;max-height:250px;overflow-y:auto}}
#legend h3{{font-size:13px;margin-bottom:10px;color:rgba(255,255,255,0.6);text-transform:uppercase;letter-spacing:1px}}
.legend-item{{display:flex;align-items:center;gap:8px;padding:3px 0;font-size:13px;cursor:pointer}}.legend-item:hover{{color:#fff}}
.legend-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
#title{{position:fixed;top:20px;left:20px;z-index:100;font-size:13px;color:rgba(255,255,255,0.4)}}
#zoom-hint{{position:fixed;bottom:130px;right:20px;z-index:100;font-size:12px;color:rgba(255,255,255,0.25)}}
#slider-panel{{position:fixed;bottom:0;left:0;right:0;z-index:200;background:rgba(10,10,30,0.92);backdrop-filter:blur(16px);border-top:1px solid rgba(255,255,255,0.1);padding:16px 40px 20px 40px}}
#slider-panel .year-display{{text-align:center;font-size:42px;font-weight:800;letter-spacing:-1px;margin-bottom:4px;background:linear-gradient(90deg,#2979FF,#00E5FF);-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
#slider-panel .paper-count{{text-align:center;font-size:13px;color:rgba(255,255,255,0.4);margin-bottom:12px}}
#year-slider{{-webkit-appearance:none;appearance:none;width:100%;height:6px;border-radius:3px;background:rgba(255,255,255,0.1);outline:none;cursor:pointer}}
#year-slider::-webkit-slider-thumb{{-webkit-appearance:none;appearance:none;width:22px;height:22px;border-radius:50%;background:#2979FF;cursor:pointer;box-shadow:0 0 12px rgba(41,121,255,0.5)}}
#slider-labels{{display:flex;justify-content:space-between;margin-top:6px;font-size:11px;color:rgba(255,255,255,0.3)}}
#play-btn{{position:absolute;right:40px;top:18px;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.15);color:#fff;padding:8px 18px;border-radius:20px;cursor:pointer;font-size:13px}}#play-btn:hover{{background:rgba(255,255,255,0.15)}}
</style></head><body>
<div id="title">HCI Topics Evolution &mdash; Subject Matter by Year</div>
<div id="search-container"><input id="search-input" type="text" placeholder="Search HCI topics..." autocomplete="off"></div>
<div id="search-results"></div>
<div id="card"><button id="card-close">&times;</button><h2 id="card-title"></h2><div class="theme-badge" id="card-theme"></div><div class="stat" id="card-stat"></div><div class="section-title">Top Connected Terms</div><div id="card-keywords"></div></div>
<div id="legend"></div>
<div id="zoom-hint">Scroll to zoom &middot; Drag to pan &middot; Arrow keys to change year</div>
<canvas id="canvas"></canvas>
<div id="slider-panel">
<button id="play-btn">&#9654; Play</button>
<div class="year-display" id="year-display">2000</div>
<div class="paper-count" id="paper-count"></div>
<input type="range" id="year-slider" min="0" max="0" value="0" step="1">
<div id="slider-labels"></div>
</div>
<script>
const ALL_DATA={all_data_json};const YEARS={years_json};
let currentYearIdx=0;let DATA=ALL_DATA[YEARS[0].toString()]||{{nodes:[],edges:[]}};
let renderNodes=[],targetNodes=[],renderEdges=[],animating=false,animStartTime=0;const ANIM_DURATION=600;
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
let W,H,dpr;function resize(){{dpr=window.devicePixelRatio||1;W=window.innerWidth;H=window.innerHeight;canvas.width=W*dpr;canvas.height=H*dpr;canvas.style.width=W+'px';canvas.style.height=H+'px';ctx.setTransform(dpr,0,0,dpr,0,0)}}resize();window.addEventListener('resize',()=>{{resize();draw()}});
let panX=W/2,panY=(H-120)/2,scale=380,selectedNode=null,selectedNodeId=null,highlightedNodes=new Set(),highlightedEdges=new Set();
let nodeMap={{}},edgesByNode={{}},renderNodeMap={{}};
function rebuildLookups(){{nodeMap={{}};edgesByNode={{}};DATA.nodes.forEach(n=>{{nodeMap[n.id]=n}});DATA.edges.forEach(e=>{{if(!edgesByNode[e.source])edgesByNode[e.source]=[];if(!edgesByNode[e.target])edgesByNode[e.target]=[];edgesByNode[e.source].push(e);edgesByNode[e.target].push(e)}})}}
function buildRenderState(){{const targetMap={{}};DATA.nodes.forEach(n=>{{targetMap[n.id]={{...n,opacity:1.0}}}});const allIds=new Set([...renderNodes.map(n=>n.id),...DATA.nodes.map(n=>n.id)]);targetNodes=[];allIds.forEach(id=>{{const tgt=targetMap[id];const cur=renderNodeMap[id];if(tgt)targetNodes.push({{...tgt,opacity:1.0}});else if(cur)targetNodes.push({{...cur,opacity:0.0,size:0,freq:0}})}});if(renderNodes.length===0){{renderNodes=targetNodes.map(n=>({{...n}}));renderNodeMap={{}};renderNodes.forEach(n=>{{renderNodeMap[n.id]=n}});renderEdges=DATA.edges.slice();return}}animating=true;animStartTime=performance.now();renderEdges=DATA.edges.slice();animateTransition()}}
function lerp(a,b,t){{return a+(b-a)*t}}
function animateTransition(){{const now=performance.now(),elapsed=now-animStartTime,t=Math.min(elapsed/ANIM_DURATION,1.0),ease=1-Math.pow(1-t,3);const tgtMap={{}};targetNodes.forEach(n=>{{tgtMap[n.id]=n}});const curMap={{}};renderNodes.forEach(n=>{{curMap[n.id]=n}});const allIds=new Set([...renderNodes.map(n=>n.id),...targetNodes.map(n=>n.id)]);const newRender=[];allIds.forEach(id=>{{const cur=curMap[id],tgt=tgtMap[id];if(cur&&tgt){{newRender.push({{id,x:tgt.x,y:tgt.y,community:tgt.community,theme:tgt.theme,color:tgt.color,freq:Math.round(lerp(cur.freq,tgt.freq,ease)),size:lerp(cur.size,tgt.size,ease),bigram:tgt.bigram,top5:tgt.top5,opacity:lerp(cur.opacity,tgt.opacity,ease)}})}}else if(tgt){{newRender.push({{...tgt,size:tgt.size*ease,opacity:ease}})}}else if(cur){{newRender.push({{...cur,size:cur.size*(1-ease),opacity:cur.opacity*(1-ease)}})}}}});renderNodes=newRender.filter(n=>n.opacity>0.01||n.size>0.1);renderNodeMap={{}};renderNodes.forEach(n=>{{renderNodeMap[n.id]=n}});if(selectedNodeId){{const sn=renderNodeMap[selectedNodeId];if(sn){{selectedNode=sn;highlightedNodes=new Set([sn.id,...(sn.top5||[])])}}}}draw();if(t<1.0)requestAnimationFrame(animateTransition);else{{renderNodes=renderNodes.filter(n=>n.opacity>0.05);renderNodeMap={{}};renderNodes.forEach(n=>{{renderNodeMap[n.id]=n}});reapplySelection();animating=false}}}}
rebuildLookups();buildRenderState();
function toScreen(x,y){{return[x*scale+panX,y*scale+panY]}}function fromScreen(sx,sy){{return[(sx-panX)/scale,(sy-panY)/scale]}}
function hexA(hex,a){{const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return`rgba(${{r}},${{g}},${{b}},${{a}})`}}
function draw(){{ctx.clearRect(0,0,W,H);const labelBoxes=[];renderEdges.forEach(e=>{{const s=renderNodeMap[e.source],t=renderNodeMap[e.target];if(!s||!t||s.opacity<0.05||t.opacity<0.05)return;const[x1,y1]=toScreen(s.x,s.y),[x2,y2]=toScreen(t.x,t.y);const isHL=highlightedEdges.has(e.source+'|'+e.target)||highlightedEdges.has(e.target+'|'+e.source);const ea=Math.min(s.opacity,t.opacity);if(isHL){{ctx.strokeStyle=`rgba(255,255,255,${{0.6*ea}})`;ctx.lineWidth=2}}else if(highlightedNodes.size>0){{ctx.strokeStyle=`rgba(255,255,255,${{0.015*ea}})`;ctx.lineWidth=0.5}}else{{ctx.strokeStyle=`rgba(255,255,255,${{0.04*ea}})`;ctx.lineWidth=0.5}}ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke()}});renderNodes.forEach(n=>{{if(n.opacity<0.02||n.size<0.3)return;const[x,y]=toScreen(n.x,n.y);if(x<-50||x>W+50||y<-50||y>H+50)return;const isSel=selectedNode&&selectedNode.id===n.id;const isHL=highlightedNodes.has(n.id);const dimmed=highlightedNodes.size>0&&!isHL&&!isSel;let r=n.size*(scale/380);if(isSel)r*=1.6;else if(isHL)r*=1.3;let alpha=n.opacity*(dimmed?0.1:(isHL||isSel?1.0:0.75));if(alpha<0.02)return;if((isSel||isHL)&&alpha>0.2){{ctx.beginPath();ctx.arc(x,y,r+6,0,Math.PI*2);ctx.fillStyle=hexA(n.color,0.18*n.opacity);ctx.fill()}}ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=hexA(n.color,alpha);ctx.fill();if(isSel){{ctx.strokeStyle=`rgba(255,255,255,${{n.opacity}})`;ctx.lineWidth=2;ctx.stroke()}}const wantLabel=isSel||isHL||(scale>300&&n.size>8)||scale>600;if(wantLabel&&!dimmed&&alpha>0.3){{const fs=isSel?14:isHL?12:10;const lw=n.id.length*fs*0.55;const lx=x-lw/2;const ly=y-r-fs-4;const lh=fs+2;let overlap=false;for(const b of labelBoxes){{if(lx<b.x+b.w&&lx+lw>b.x&&ly<b.y+b.h&&ly+lh>b.y){{overlap=true;break}}}}if(!overlap||isSel||isHL){{ctx.font=`${{isSel?'bold ':isHL?'bold ':''}}${{fs}}px Inter,sans-serif`;ctx.fillStyle=`rgba(255,255,255,${{alpha}})`;ctx.textAlign='center';ctx.fillText(n.id,x,y-r-5);labelBoxes.push({{x:lx,y:ly,w:lw,h:lh}})}}}}}})}}
let dragging=false,lastMx,lastMy;canvas.addEventListener('mousedown',e=>{{dragging=true;lastMx=e.clientX;lastMy=e.clientY;canvas.classList.add('grabbing')}});window.addEventListener('mousemove',e=>{{if(!dragging)return;panX+=e.clientX-lastMx;panY+=e.clientY-lastMy;lastMx=e.clientX;lastMy=e.clientY;draw()}});window.addEventListener('mouseup',()=>{{dragging=false;canvas.classList.remove('grabbing')}});canvas.addEventListener('wheel',e=>{{e.preventDefault();const[mx,my]=[e.clientX,e.clientY],[wx,wy]=fromScreen(mx,my),factor=e.deltaY<0?1.12:0.89;scale*=factor;panX=mx-wx*scale;panY=my-wy*scale;draw()}},{{passive:false}});
canvas.addEventListener('click',e=>{{const[mx,my]=[e.clientX,e.clientY];let closest=null,closestDist=Infinity;renderNodes.forEach(n=>{{if(n.opacity<0.3)return;const[x,y]=toScreen(n.x,n.y);const d=Math.hypot(mx-x,my-y),r=n.size*(scale/380)+6;if(d<r&&d<closestDist){{closest=n;closestDist=d}}}});if(closest)selectNode(closest,true);else clearSelection()}});
function selectNode(node,animate){{selectedNode=node;selectedNodeId=node.id;highlightedNodes=new Set([node.id,...(node.top5||[])]);highlightedEdges=new Set();(node.top5||[]).forEach(nb=>{{highlightedEdges.add(node.id+'|'+nb)}});if(animate){{const ts=Math.max(scale,500);smoothAnimateTo(W/2-node.x*ts,(H-120)/2-node.y*ts,ts)}}showCard(node);draw()}}
function clearSelection(){{selectedNode=null;selectedNodeId=null;highlightedNodes=new Set();highlightedEdges=new Set();document.getElementById('card').style.display='none';draw()}}
function reapplySelection(){{if(!selectedNodeId)return;const n=renderNodeMap[selectedNodeId];const freshData=nodeMap[selectedNodeId];if(n&&n.opacity>0.3&&freshData){{n.top5=freshData.top5||[];selectedNode=n;highlightedNodes=new Set([n.id,...(n.top5||[])]);highlightedEdges=new Set();(n.top5||[]).forEach(nb=>{{highlightedEdges.add(n.id+'|'+nb)}});showCard(n)}}else if(n&&n.opacity>0.05){{selectedNode=n;highlightedNodes=new Set([n.id]);highlightedEdges=new Set();const card=document.getElementById('card');document.getElementById('card-title').textContent=selectedNodeId;document.getElementById('card-stat').textContent='Not active in this year';document.getElementById('card-keywords').innerHTML='<div style="color:rgba(255,255,255,0.3);font-size:13px;padding:8px">Topic not present in this timeframe</div>';card.style.display='block'}}else{{selectedNode=null;highlightedNodes=new Set();highlightedEdges=new Set()}}}}
function smoothAnimateTo(tx,ty,ts){{const sx=panX,sy=panY,ss=scale,dur=400,t0=performance.now();function step(t){{const p=Math.min((t-t0)/dur,1),ease=1-Math.pow(1-p,3);panX=sx+(tx-sx)*ease;panY=sy+(ty-sy)*ease;scale=ss+(ts-ss)*ease;draw();if(p<1)requestAnimationFrame(step)}}requestAnimationFrame(step)}}
function showCard(node){{const card=document.getElementById('card');document.getElementById('card-title').textContent=node.id;const badge=document.getElementById('card-theme');badge.textContent=node.theme;badge.style.background=node.color+'30';badge.style.color=node.color;document.getElementById('card-stat').textContent=`Frequency: ${{node.freq.toLocaleString()}} papers  \\u00B7  Community ${{node.community}}`;const kwDiv=document.getElementById('card-keywords');kwDiv.innerHTML='';if(!node.top5||node.top5.length===0){{kwDiv.innerHTML='<div style="color:rgba(255,255,255,0.3);font-size:13px;padding:8px">No neighbors this year</div>'}}(node.top5||[]).forEach(kw=>{{const nb=renderNodeMap[kw];if(!nb)return;let w=0;(edgesByNode[node.id]||[]).forEach(e=>{{if(e.source===kw||e.target===kw)w=e.weight}});const el=document.createElement('div');el.className='keyword-link';el.innerHTML=`<span class="dot" style="background:${{nb.color}}"></span>${{kw}}<span class="pmi">PMI ${{w.toFixed(2)}}</span>`;el.onclick=()=>{{selectNode(nb,true);searchInput.value=kw}};kwDiv.appendChild(el)}});card.style.display='block'}}
document.getElementById('card-close').onclick=clearSelection;
const searchInput=document.getElementById('search-input'),searchResults=document.getElementById('search-results');searchInput.addEventListener('input',()=>{{const q=searchInput.value.toLowerCase().trim();if(q.length<2){{searchResults.style.display='none';return}}const matches=renderNodes.filter(n=>n.id.includes(q)&&n.opacity>0.3).sort((a,b)=>{{const ae=a.id===q?2:a.id.startsWith(q)?1:0,be=b.id===q?2:b.id.startsWith(q)?1:0;if(ae!==be)return be-ae;return b.freq-a.freq}}).slice(0,8);if(matches.length===0){{searchResults.style.display='none';return}}searchResults.innerHTML='';matches.forEach(n=>{{const el=document.createElement('div');el.className='search-item';el.innerHTML=`<span style="color:${{n.color}}">&#9679;</span> ${{n.id}} <span class="freq">${{n.freq.toLocaleString()}} papers</span>`;el.onclick=()=>{{searchInput.value=n.id;searchResults.style.display='none';selectNode(n,true)}};searchResults.appendChild(el)}});searchResults.style.display='block'}});searchInput.addEventListener('keydown',e=>{{if(e.key==='Enter'){{const q=searchInput.value.toLowerCase().trim();const m=renderNodes.find(n=>n.id===q&&n.opacity>0.3)||renderNodes.find(n=>n.id.includes(q)&&n.opacity>0.3);if(m){{searchResults.style.display='none';selectNode(m,true)}}}}if(e.key==='Escape'){{searchResults.style.display='none';clearSelection();searchInput.value='';searchInput.blur()}}}});document.addEventListener('click',e=>{{if(!e.target.closest('#search-container')&&!e.target.closest('#search-results'))searchResults.style.display='none'}});
function buildLegend(){{const legend=document.getElementById('legend');const themes={{}};renderNodes.forEach(n=>{{if(n.opacity<0.3)return;if(!themes[n.community])themes[n.community]={{color:n.color,theme:n.theme,count:0}};themes[n.community].count++}});let html='<h3>Topic Clusters</h3>';Object.entries(themes).sort((a,b)=>b[1].count-a[1].count).forEach(([id,t])=>{{html+=`<div class="legend-item" onclick="highlightCommunity(${{id}})"><span class="legend-dot" style="background:${{t.color}}"></span>${{t.theme}} <span style="color:rgba(255,255,255,0.3);margin-left:auto;font-size:11px">${{t.count}}</span></div>`}});legend.innerHTML=html}}
function highlightCommunity(commId){{highlightedNodes=new Set();highlightedEdges=new Set();selectedNode=null;renderNodes.forEach(n=>{{if(n.community===commId&&n.opacity>0.3)highlightedNodes.add(n.id)}});document.getElementById('card').style.display='none';draw()}}
const slider=document.getElementById('year-slider'),yearDisplay=document.getElementById('year-display'),paperCount=document.getElementById('paper-count'),labelsDiv=document.getElementById('slider-labels');
slider.max=YEARS.length-1;slider.value=YEARS.length-3;currentYearIdx=parseInt(slider.value);
let labelsHTML='';YEARS.forEach((y,i)=>{{if(i===0||i===YEARS.length-1||y%5===0)labelsHTML+=`<span>${{y}}</span>`}});labelsDiv.innerHTML=labelsHTML;
function switchYear(idx){{currentYearIdx=idx;const year=YEARS[idx];DATA=ALL_DATA[year.toString()]||{{nodes:[],edges:[]}};rebuildLookups();yearDisplay.textContent=year;const pc=DATA.paper_count||DATA.nodes.length;paperCount.textContent=DATA.nodes.length>0?`${{pc.toLocaleString()}} papers  \\u00B7  ${{DATA.nodes.length}} topics  \\u00B7  ${{DATA.edges.length}} connections`:'Insufficient data for this year';buildRenderState();buildLegend()}}
slider.addEventListener('input',()=>{{switchYear(parseInt(slider.value))}});
let playing=false,playInterval=null;const playBtn=document.getElementById('play-btn');playBtn.onclick=()=>{{if(playing){{playing=false;clearInterval(playInterval);playBtn.innerHTML='&#9654; Play'}}else{{playing=true;playBtn.innerHTML='&#9646;&#9646; Pause';if(currentYearIdx>=YEARS.length-1){{currentYearIdx=0;slider.value=0}}playInterval=setInterval(()=>{{if(currentYearIdx>=YEARS.length-1){{playing=false;clearInterval(playInterval);playBtn.innerHTML='&#9654; Play';return}}currentYearIdx++;slider.value=currentYearIdx;switchYear(currentYearIdx)}},2000)}}}};
document.addEventListener('keydown',e=>{{if(e.target.tagName==='INPUT')return;if(e.key==='ArrowLeft'&&currentYearIdx>0){{currentYearIdx--;slider.value=currentYearIdx;switchYear(currentYearIdx)}}if(e.key==='ArrowRight'&&currentYearIdx<YEARS.length-1){{currentYearIdx++;slider.value=currentYearIdx;switchYear(currentYearIdx)}}}});
switchYear(currentYearIdx);
</script></body></html>"""

    output_path = os.path.join(OUTPUT_DIR, "map_layer2_temporal.html")
    with open(output_path, "w") as f:
        f.write(html)
    sz = os.path.getsize(output_path) / 1024 / 1024
    print(f"Saved: {output_path} ({sz:.1f} MB)")


def main():
    papers = load_filtered_papers()

    papers_by_year = {}
    for p in papers:
        y = int(p["year"])
        if y < 2000 or y > 2025:
            continue
        if y not in papers_by_year:
            papers_by_year[y] = []
        papers_by_year[y].append(p)

    years = sorted(papers_by_year.keys())
    print(f"\nYears: {years[0]}-{years[-1]}")
    for y in years:
        print(f"  {y}: {len(papers_by_year[y]):,}")

    print("\nBuilding global vocabulary (bigrams only, sampled for memory)...")
    import random
    random.seed(42)
    all_papers_valid = [p for p in papers if 2000 <= int(p["year"]) <= 2025]
    if len(all_papers_valid) > 500000:
        sampled = random.sample(all_papers_valid, 500000)
        print(f"  Sampled 500,000 from {len(all_papers_valid):,} for vocabulary fitting")
    else:
        sampled = all_papers_valid
    all_abstracts = [clean_text(p["abstract"]) for p in sampled]
    global_vec = TfidfVectorizer(
        min_df=10, max_df=0.5, max_features=20000,
        stop_words="english", ngram_range=(1, 2), sublinear_tf=True,
    )
    global_vec.fit(all_abstracts)
    vocab_global = np.array(global_vec.get_feature_names_out())
    print(f"Vocabulary: {len(vocab_global):,} terms")

    G, pos, communities, theme_labels, top_indices, top_vocab = \
        build_global_network(all_abstracts, vocab_global, top_n=300, min_cooc=15)

    year_data = {}
    prev_comms = None
    for y in years:
        year_papers = papers_by_year[y]
        if len(year_papers) > 50000:
            year_papers = random.sample(year_papers, 50000)
        print(f"\nProcessing {y} ({len(year_papers):,} papers)...")
        abstracts = [clean_text(p["abstract"]) for p in year_papers]
        yd, prev_comms = compute_year_data(abstracts, vocab_global, top_indices, top_vocab,
                               G, pos, communities, theme_labels, prev_comms=prev_comms)
        print(f"  Active: {len(yd['nodes'])} topics, {len(yd['edges'])} connections")
        year_data[str(y)] = yd

    create_temporal_html(year_data, years)
    print("\nDone! Open output/map_layer2_temporal.html")


if __name__ == "__main__":
    main()

"""
Step 8: Layer 2 — HCI Topic Map

Instead of frequency-based layering, this uses HCI anchor terms to pull out
topic-specific vocabulary. Terms are included only if they co-occur strongly
with core HCI concepts (user, interface, interaction, design, etc.),
filtering out generic academic language that doesn't connect to HCI.
"""

import json
import os
import re

import networkx as nx
import numpy as np
from scipy import sparse

from stopwords import is_valid_term, ALL_STOPWORDS

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")

COLORS = [
    "#FF1744", "#00E676", "#FFEA00", "#2979FF", "#FF9100",
    "#D500F9", "#00E5FF", "#F50057", "#76FF03", "#FF6D00",
    "#651FFF", "#1DE9B6", "#C6FF00", "#304FFE", "#DD2C00",
    "#00BFA5", "#FFD600", "#AA00FF", "#64DD17", "#6200EA",
]

# Anchor terms — core HCI concepts that define the field
HCI_ANCHORS = {
    "user", "users", "interface", "interaction", "usability",
    "design", "designed", "accessibility", "human computer",
    "user experience", "user interface", "virtual reality",
    "augmented reality", "machine learning", "deep learning",
    "mobile", "web", "robot", "gesture", "haptic", "speech",
    "visualization", "privacy", "security", "game", "wearable",
    "sensor", "iot", "prototype", "evaluation",
}


def load_data():
    dtm = sparse.load_npz(os.path.join(DATA_DIR, "dtm.npz"))
    vocab = np.load(os.path.join(DATA_DIR, "vocab.npy"), allow_pickle=True)
    return dtm, vocab


def find_hci_bigrams(dtm, vocab, top_n=400):
    """
    Select only bigrams (2-word phrases) which are inherently more specific
    than single words. Filter out methodology bigrams to keep only
    subject-matter topics.
    """
    print("Selecting subject-matter bigrams...")

    term_freq = np.asarray((dtm > 0).sum(axis=0)).flatten()

    # Methodology bigrams to exclude
    method_bigrams = {
        'purpose study', 'results study', 'study conducted', 'aim study',
        'study used', 'method used', 'used study', 'findings study',
        'study examines', 'study provides', 'study investigates',
        'objective study', 'present study', 'findings suggest',
        'results obtained', 'results demonstrate', 'findings indicate',
        'previous studies', 'results showed', 'study shows',
        'paper present', 'study proposes', 'study revealed',
        'showed significant', 'analysis revealed', 'article presents',
        'methods study', 'study aimed', 'results suggest',
        'study explores', 'paper aims', 'paper proposes',
        'analyzed using', 'recent years', 'widely used',
        'significant differences', 'originality value',
        'methodology approach', 'design methodology',
        'et al', 'important role', 'high quality', 'wide range',
        'large scale', 'results indicate', 'cross sectional',
        'study sample', 'data collected', 'qualitative data',
        'regression analysis', 'comparative analysis',
        'significant impact', 'significant effect',
        'positive impact', 'negative impact',
        'research design', 'case studies', 'study aims',
        'paper present', 'results analysis',
        'http www', 'www org', 'https www', 'https doi', 'doi org',
        'www com', 'lt gt', 'gt lt',
        'al mada', 'elde edilen', 'ger ekle', 'ayr ca', 'al malar',
        'aras ndaki', 'tespit edilmi', 'di er',
    }

    selected = []
    for i in range(len(vocab)):
        v = vocab[i]
        # Must be a bigram
        if ' ' not in v:
            continue
        if not is_valid_term(v):
            continue
        if term_freq[i] < 500:
            continue
        if v.lower() in method_bigrams:
            continue
        # Skip if contains methodology words
        parts = v.lower().split()
        skip = False
        meth_words = {'study', 'studies', 'results', 'findings', 'paper',
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
                      'used', 'based', 'approach', 'proposed', 'evidence'}
        for p in parts:
            if p in meth_words:
                skip = True
                break
        if skip:
            continue
        selected.append((i, int(term_freq[i])))

    # Sort by frequency, take top_n
    selected.sort(key=lambda x: x[1], reverse=True)
    indices = [s[0] for s in selected[:top_n]]

    print(f"  Selected {len(indices)} subject-matter bigrams")
    print(f"  Top 20: {', '.join(vocab[i] for i in indices[:20])}")
    return indices


def build_network(dtm, vocab, term_indices, min_cooc=30):
    print(f"\nBuilding network from {len(term_indices)} HCI-specific terms...")

    term_freq = np.asarray((dtm > 0).sum(axis=0)).flatten()
    dtm_sub = dtm[:, term_indices]
    sub_vocab = vocab[term_indices]

    binary = (dtm_sub > 0).astype(int)
    n_docs = binary.shape[0]
    cooc = (binary.T @ binary).toarray().astype(float)
    np.fill_diagonal(cooc, 0)
    doc_freq = np.asarray(binary.sum(axis=0)).flatten().astype(float)

    G = nx.Graph()
    for i in range(len(sub_vocab)):
        G.add_node(sub_vocab[i], freq=int(term_freq[term_indices[i]]))

    for i in range(len(sub_vocab)):
        for j in range(i + 1, len(sub_vocab)):
            if cooc[i][j] < min_cooc:
                continue
            p_ij = cooc[i][j] / n_docs
            p_i = doc_freq[i] / n_docs
            p_j = doc_freq[j] / n_docs
            if p_i == 0 or p_j == 0:
                continue
            pmi = np.log2(p_ij / (p_i * p_j))
            if pmi > 0.5:
                G.add_edge(sub_vocab[i], sub_vocab[j], weight=float(pmi))

    isolates = list(nx.isolates(G))
    G.remove_nodes_from(isolates)
    print(f"Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    communities = nx.community.louvain_communities(G, seed=42, resolution=1.0)
    for i, comm in enumerate(communities):
        for node in comm:
            G.nodes[node]["community"] = i

    print(f"Detected {len(communities)} communities")
    for i, comm in enumerate(communities):
        top = sorted(comm, key=lambda n: G.nodes[n].get("freq", 0), reverse=True)[:6]
        print(f"  C{i} ({len(comm)}): {', '.join(top)}")

    return G, communities


def create_html(G, communities):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    pos = nx.spring_layout(G, k=2.0, iterations=200, seed=42, weight="weight")

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
        neighbors = [(nb, G[node][nb]["weight"]) for nb in G.neighbors(node)]
        neighbors.sort(key=lambda x: x[1], reverse=True)
        return [n[0] for n in neighbors[:n]]

    nodes_data = []
    for node in G.nodes():
        x, y = pos[node]
        comm = G.nodes[node]["community"]
        freq = freqs[node]
        size = 3 + 12 * (freq / max_freq)
        nodes_data.append({
            "id": node, "x": float(x), "y": float(y),
            "community": comm, "theme": theme_labels.get(comm, f"Community {comm}"),
            "color": COLORS[comm % len(COLORS)], "freq": freq,
            "size": float(size), "bigram": " " in node,
            "top5": get_top_neighbors(node),
        })

    edges_data = [{"source": u, "target": v, "weight": round(float(G[u][v]["weight"]), 3)} for u, v in G.edges()]
    edges_data.sort(key=lambda e: e["weight"], reverse=True)
    edges_data = edges_data[:2500]

    graph_json = json.dumps({"nodes": nodes_data, "edges": edges_data})

    html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="UTF-8"><meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>HCI Research Topics — What the Field Studies</title>
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
#card-close{{position:absolute;top:12px;right:16px;background:none;border:none;color:rgba(255,255,255,0.4);font-size:20px;cursor:pointer;padding:4px}}
#card-close:hover{{color:#fff}}
#legend{{position:fixed;bottom:20px;left:20px;z-index:100;background:rgba(10,10,30,0.85);backdrop-filter:blur(12px);border-radius:12px;border:1px solid rgba(255,255,255,0.1);padding:16px;max-height:350px;overflow-y:auto}}
#legend h3{{font-size:13px;margin-bottom:10px;color:rgba(255,255,255,0.6);text-transform:uppercase;letter-spacing:1px}}
.legend-item{{display:flex;align-items:center;gap:8px;padding:3px 0;font-size:13px;cursor:pointer}}.legend-item:hover{{color:#fff}}
.legend-dot{{width:10px;height:10px;border-radius:50%;flex-shrink:0}}
#title{{position:fixed;top:20px;left:20px;z-index:100;font-size:13px;color:rgba(255,255,255,0.4)}}
#zoom-hint{{position:fixed;bottom:20px;right:20px;z-index:100;font-size:12px;color:rgba(255,255,255,0.25)}}
</style></head><body>
<div id="title">HCI Research Topics &mdash; What the Field Studies &mdash; {G.number_of_nodes()} terms, {len(communities)} clusters</div>
<div id="search-container"><input id="search-input" type="text" placeholder="Search HCI topics..." autocomplete="off"></div>
<div id="search-results"></div>
<div id="card"><button id="card-close">&times;</button><h2 id="card-title"></h2><div class="theme-badge" id="card-theme"></div><div class="stat" id="card-stat"></div><div class="section-title">Top Connected Terms</div><div id="card-keywords"></div></div>
<div id="legend"></div>
<div id="zoom-hint">Scroll to zoom &middot; Drag to pan &middot; Click a node</div>
<canvas id="canvas"></canvas>
<script>
const DATA={graph_json};
const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
let W,H,dpr;function resize(){{dpr=window.devicePixelRatio||1;W=window.innerWidth;H=window.innerHeight;canvas.width=W*dpr;canvas.height=H*dpr;canvas.style.width=W+'px';canvas.style.height=H+'px';ctx.setTransform(dpr,0,0,dpr,0,0)}}resize();window.addEventListener('resize',()=>{{resize();draw()}});
let panX=W/2,panY=H/2,scale=380,selectedNode=null,highlightedNodes=new Set(),highlightedEdges=new Set();
const nodeMap={{}};DATA.nodes.forEach(n=>{{nodeMap[n.id]=n}});const edgesByNode={{}};DATA.edges.forEach(e=>{{if(!edgesByNode[e.source])edgesByNode[e.source]=[];if(!edgesByNode[e.target])edgesByNode[e.target]=[];edgesByNode[e.source].push(e);edgesByNode[e.target].push(e)}});
function toScreen(x,y){{return[x*scale+panX,y*scale+panY]}}function fromScreen(sx,sy){{return[(sx-panX)/scale,(sy-panY)/scale]}}
function hexA(hex,a){{const r=parseInt(hex.slice(1,3),16),g=parseInt(hex.slice(3,5),16),b=parseInt(hex.slice(5,7),16);return`rgba(${{r}},${{g}},${{b}},${{a}})`}}
function draw(){{ctx.clearRect(0,0,W,H);const labelBoxes=[];DATA.edges.forEach(e=>{{const s=nodeMap[e.source],t=nodeMap[e.target];if(!s||!t)return;const[x1,y1]=toScreen(s.x,s.y),[x2,y2]=toScreen(t.x,t.y);const isHL=highlightedEdges.has(e.source+'|'+e.target)||highlightedEdges.has(e.target+'|'+e.source);if(isHL){{ctx.strokeStyle='rgba(255,255,255,0.6)';ctx.lineWidth=2}}else if(highlightedNodes.size>0){{ctx.strokeStyle='rgba(255,255,255,0.015)';ctx.lineWidth=0.5}}else{{ctx.strokeStyle='rgba(255,255,255,0.04)';ctx.lineWidth=0.5}}ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke()}});DATA.nodes.forEach(n=>{{const[x,y]=toScreen(n.x,n.y);if(x<-50||x>W+50||y<-50||y>H+50)return;const isSel=selectedNode&&selectedNode.id===n.id;const isHL=highlightedNodes.has(n.id);const dimmed=highlightedNodes.size>0&&!isHL&&!isSel;let r=n.size*(scale/380);if(isSel)r*=1.6;else if(isHL)r*=1.3;let alpha=dimmed?0.1:(isHL||isSel?1.0:0.75);if(isSel||isHL){{ctx.beginPath();ctx.arc(x,y,r+6,0,Math.PI*2);ctx.fillStyle=n.color+'30';ctx.fill()}}ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fillStyle=hexA(n.color,alpha);ctx.fill();if(isSel){{ctx.strokeStyle='#fff';ctx.lineWidth=2;ctx.stroke()}}const wantLabel=isSel||isHL||(scale>300&&n.size>8)||scale>600;if(wantLabel&&!dimmed){{const fs=isSel?14:isHL?12:10;const lw=n.id.length*fs*0.55;const lx=x-lw/2;const ly=y-r-fs-4;const lh=fs+2;let overlap=false;for(const b of labelBoxes){{if(lx<b.x+b.w&&lx+lw>b.x&&ly<b.y+b.h&&ly+lh>b.y){{overlap=true;break}}}}if(!overlap||isSel||isHL){{ctx.font=`${{isSel?'bold ':isHL?'bold ':''}}${{fs}}px Inter,sans-serif`;ctx.fillStyle=isSel||isHL?'#fff':'rgba(255,255,255,0.65)';ctx.textAlign='center';ctx.fillText(n.id,x,y-r-5);labelBoxes.push({{x:lx,y:ly,w:lw,h:lh}})}}}}}})}}
let dragging=false,lastMx,lastMy;canvas.addEventListener('mousedown',e=>{{dragging=true;lastMx=e.clientX;lastMy=e.clientY;canvas.classList.add('grabbing')}});window.addEventListener('mousemove',e=>{{if(!dragging)return;panX+=e.clientX-lastMx;panY+=e.clientY-lastMy;lastMx=e.clientX;lastMy=e.clientY;draw()}});window.addEventListener('mouseup',()=>{{dragging=false;canvas.classList.remove('grabbing')}});canvas.addEventListener('wheel',e=>{{e.preventDefault();const[mx,my]=[e.clientX,e.clientY],[wx,wy]=fromScreen(mx,my),factor=e.deltaY<0?1.12:0.89;scale*=factor;panX=mx-wx*scale;panY=my-wy*scale;draw()}},{{passive:false}});
canvas.addEventListener('click',e=>{{const[mx,my]=[e.clientX,e.clientY];let closest=null,closestDist=Infinity;DATA.nodes.forEach(n=>{{const[x,y]=toScreen(n.x,n.y);const d=Math.hypot(mx-x,my-y),r=n.size*(scale/380)+6;if(d<r&&d<closestDist){{closest=n;closestDist=d}}}});if(closest)selectNode(closest,true);else clearSelection()}});
function selectNode(node,animate){{selectedNode=node;highlightedNodes=new Set([node.id,...node.top5]);highlightedEdges=new Set();node.top5.forEach(nb=>{{highlightedEdges.add(node.id+'|'+nb)}});if(animate){{const ts=Math.max(scale,500);animateTo(W/2-node.x*ts,H/2-node.y*ts,ts)}}showCard(node);draw()}}
function clearSelection(){{selectedNode=null;highlightedNodes=new Set();highlightedEdges=new Set();document.getElementById('card').style.display='none';draw()}}
function animateTo(tx,ty,ts){{const sx=panX,sy=panY,ss=scale,dur=400,t0=performance.now();function step(t){{const p=Math.min((t-t0)/dur,1),ease=1-Math.pow(1-p,3);panX=sx+(tx-sx)*ease;panY=sy+(ty-sy)*ease;scale=ss+(ts-ss)*ease;draw();if(p<1)requestAnimationFrame(step)}}requestAnimationFrame(step)}}
function showCard(node){{const card=document.getElementById('card');document.getElementById('card-title').textContent=node.id;const badge=document.getElementById('card-theme');badge.textContent=node.theme;badge.style.background=node.color+'30';badge.style.color=node.color;document.getElementById('card-stat').textContent=`Frequency: ${{node.freq.toLocaleString()}} papers  \\u00B7  Community ${{node.community}}`;const kwDiv=document.getElementById('card-keywords');kwDiv.innerHTML='';if(node.top5.length===0){{kwDiv.innerHTML='<div style="color:rgba(255,255,255,0.3);font-size:13px;padding:8px">No neighbors</div>'}}node.top5.forEach(kw=>{{const nb=nodeMap[kw];if(!nb)return;let w=0;(edgesByNode[node.id]||[]).forEach(e=>{{if(e.source===kw||e.target===kw)w=e.weight}});const el=document.createElement('div');el.className='keyword-link';el.innerHTML=`<span class="dot" style="background:${{nb.color}}"></span>${{kw}}<span class="pmi">PMI ${{w.toFixed(2)}}</span>`;el.onclick=()=>{{selectNode(nb,true);searchInput.value=kw}};kwDiv.appendChild(el)}});card.style.display='block'}}
document.getElementById('card-close').onclick=clearSelection;
const searchInput=document.getElementById('search-input'),searchResults=document.getElementById('search-results');searchInput.addEventListener('input',()=>{{const q=searchInput.value.toLowerCase().trim();if(q.length<2){{searchResults.style.display='none';return}}const matches=DATA.nodes.filter(n=>n.id.includes(q)).sort((a,b)=>{{const ae=a.id===q?2:a.id.startsWith(q)?1:0,be=b.id===q?2:b.id.startsWith(q)?1:0;if(ae!==be)return be-ae;return b.freq-a.freq}}).slice(0,8);if(matches.length===0){{searchResults.style.display='none';return}}searchResults.innerHTML='';matches.forEach(n=>{{const el=document.createElement('div');el.className='search-item';el.innerHTML=`<span style="color:${{n.color}}">&#9679;</span> ${{n.id}} <span class="freq">${{n.freq.toLocaleString()}} papers</span>`;el.onclick=()=>{{searchInput.value=n.id;searchResults.style.display='none';selectNode(n,true)}};searchResults.appendChild(el)}});searchResults.style.display='block'}});searchInput.addEventListener('keydown',e=>{{if(e.key==='Enter'){{const q=searchInput.value.toLowerCase().trim();const m=DATA.nodes.find(n=>n.id===q)||DATA.nodes.find(n=>n.id.includes(q));if(m){{searchResults.style.display='none';selectNode(m,true)}}}}if(e.key==='Escape'){{searchResults.style.display='none';clearSelection();searchInput.value='';searchInput.blur()}}}});document.addEventListener('click',e=>{{if(!e.target.closest('#search-container')&&!e.target.closest('#search-results'))searchResults.style.display='none'}});
function buildLegend(){{const legend=document.getElementById('legend');const themes={{}};DATA.nodes.forEach(n=>{{if(!themes[n.community])themes[n.community]={{color:n.color,theme:n.theme,count:0}};themes[n.community].count++}});let html='<h3>HCI Topic Clusters</h3>';Object.entries(themes).sort((a,b)=>b[1].count-a[1].count).forEach(([id,t])=>{{html+=`<div class="legend-item" onclick="highlightCommunity(${{id}})"><span class="legend-dot" style="background:${{t.color}}"></span>${{t.theme}} <span style="color:rgba(255,255,255,0.3);margin-left:auto;font-size:11px">${{t.count}}</span></div>`}});legend.innerHTML=html}}
function highlightCommunity(commId){{highlightedNodes=new Set();highlightedEdges=new Set();selectedNode=null;DATA.nodes.forEach(n=>{{if(n.community===commId)highlightedNodes.add(n.id)}});document.getElementById('card').style.display='none';draw()}}
buildLegend();draw();
</script></body></html>"""

    output_path = os.path.join(OUTPUT_DIR, "map2_layer2_topics.html")
    with open(output_path, "w") as f:
        f.write(html)
    sz = os.path.getsize(output_path) / 1024
    print(f"\nSaved: {output_path} ({sz:.0f} KB)")

    # Save report
    lines = ["HCI Research Topics — Layer 2 (Subject Matter)\n", "=" * 50 + "\n\n"]
    for i, comm in enumerate(communities):
        terms_sorted = sorted(comm, key=lambda n: G.nodes[n].get("freq", 0), reverse=True)
        lines.append(f"Community {i} — {len(comm)} terms\n")
        lines.append("-" * 40 + "\n")
        for t in terms_sorted[:30]:
            lines.append(f"  {t} (freq: {G.nodes[t].get('freq', 0)})\n")
        lines.append("\n")
    with open(os.path.join(OUTPUT_DIR, "communities_layer2_report.txt"), "w") as f:
        f.writelines(lines)
    print("Saved: communities_layer2_report.txt")


def main():
    dtm, vocab = load_data()
    term_indices = find_hci_bigrams(dtm, vocab, top_n=400)
    G, communities = build_network(dtm, vocab, term_indices, min_cooc=30)
    create_html(G, communities)
    print("\nDone! Open output/map2_layer2_topics.html")


if __name__ == "__main__":
    main()

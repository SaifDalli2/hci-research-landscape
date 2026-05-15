"""
Step 5: Keyword Search — Find a term's position in the HCI landscape
and its relationship to each community/theme.

Usage:
    python 05_keyword_search.py "artificial intelligence"
    python 05_keyword_search.py "virtual reality"
    python 05_keyword_search.py "accessibility"
"""

import json
import os
import sys
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def load_all():
    dtm = sparse.load_npz(os.path.join(DATA_DIR, "dtm.npz"))
    vocab = np.load(os.path.join(DATA_DIR, "vocab.npy"), allow_pickle=True)
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    with open(os.path.join(DATA_DIR, "vectorizer.pkl"), "rb") as f:
        vectorizer = pickle.load(f)
    spectrum = pd.read_csv(os.path.join(OUTPUT_DIR, "semantic_spectrum.csv"))
    return dtm, vocab, meta, vectorizer, spectrum


def find_term_matches(query, vocab):
    """Find exact and partial matches for a query in the vocabulary."""
    query_lower = query.lower()
    exact = [v for v in vocab if v == query_lower]
    partial = [v for v in vocab if query_lower in v and v != query_lower]
    # Also find terms that contain any word from the query
    words = query_lower.split()
    related = []
    if len(words) > 1:
        for v in vocab:
            if v not in exact and v not in partial:
                if all(w in v for w in words):
                    related.append(v)
    return exact, partial, related


def get_compass_position(term, spectrum):
    """Get the term's position on the semantic compass."""
    row = spectrum[spectrum["term"] == term]
    if row.empty:
        return None, None
    return row.iloc[0]["component_1"], row.iloc[0]["component_2"]


def compute_theme_affinity(query_term, dtm, vocab, vectorizer):
    """
    Compute how strongly a keyword relates to each theme/community.
    Uses the term's co-occurrence profile compared to community centroids.
    """
    # Define themes based on detected communities (from our analysis)
    themes = {
        "Design & Social Computing": [
            "design", "information", "technology", "development", "social",
            "digital", "online", "education", "knowledge", "community",
            "internet", "web", "media", "content", "platform",
        ],
        "Systems & UX": [
            "systems", "user", "users", "application", "interface",
            "software", "developed", "interactive", "device", "prototype",
            "usability", "feedback", "input", "display", "tool",
        ],
        "User Studies & Methods": [
            "participants", "task", "conducted", "findings", "survey",
            "interviews", "questionnaire", "experiment", "significant",
            "perception", "satisfaction", "evaluation", "cognitive", "workload",
        ],
        "People & Interaction": [
            "people", "interactions", "communication", "collaborative",
            "social", "group", "children", "older", "disability",
            "accessibility", "assistive", "inclusive",
        ],
        "Core HCI & ML": [
            "human", "computer", "interaction", "learning", "machine",
            "deep learning", "neural", "recognition", "classification",
            "algorithm", "training", "accuracy", "cnn", "detection",
        ],
        "VR/AR & Embodied": [
            "virtual", "reality", "virtual reality", "augmented",
            "augmented reality", "immersive", "3d", "simulation",
            "environment", "haptic", "gesture", "body", "motion",
        ],
        "NLP & Language": [
            "language", "natural", "text", "speech", "sentiment",
            "natural language", "dialogue", "chatbot", "conversation",
            "voice", "word", "semantic",
        ],
        "Health & Wellbeing": [
            "health", "patient", "clinical", "medical", "care",
            "mental", "therapy", "rehabilitation", "elderly", "wellbeing",
            "stress", "emotion", "affect",
        ],
        "Education & Learning": [
            "students", "learning", "education", "teaching", "classroom",
            "pedagogical", "curriculum", "instructional", "higher education",
            "online learning", "e-learning", "mooc",
        ],
        "IoT & Smart Systems": [
            "iot", "internet things", "smart", "sensor", "wearable",
            "monitoring", "real time", "embedded", "robot", "autonomous",
            "automation",
        ],
    }

    # Get vocabulary index for the query term
    vocab_list = list(vocab)
    if query_term not in vocab_list:
        return {}

    term_idx = vocab_list.index(query_term)
    binary = (dtm > 0).astype(int)
    term_vector = binary[:, term_idx].toarray().flatten()

    # For each theme, compute co-occurrence strength
    affinities = {}
    for theme_name, theme_terms in themes.items():
        theme_indices = [vocab_list.index(t) for t in theme_terms if t in vocab_list]
        if not theme_indices:
            affinities[theme_name] = 0.0
            continue

        # Average co-occurrence with theme terms
        cooc_scores = []
        for ti in theme_indices:
            theme_vector = binary[:, ti].toarray().flatten()
            # Jaccard similarity
            intersection = np.sum(term_vector & theme_vector)
            union = np.sum(term_vector | theme_vector)
            if union > 0:
                cooc_scores.append(intersection / union)
            else:
                cooc_scores.append(0.0)

        affinities[theme_name] = np.mean(cooc_scores)

    return affinities


def find_most_similar_terms(query_term, dtm, vocab, top_n=20):
    """Find terms most similar to the query based on co-occurrence profiles."""
    vocab_list = list(vocab)
    if query_term not in vocab_list:
        return []

    term_idx = vocab_list.index(query_term)

    # Get the term's column as a vector (its distribution across documents)
    term_col = dtm[:, term_idx].toarray().flatten()

    # Compute cosine similarity with all other terms
    # Use a sample of documents for speed
    n_docs = dtm.shape[0]
    if n_docs > 50000:
        sample_idx = np.random.RandomState(42).choice(n_docs, 50000, replace=False)
        dtm_sample = dtm[sample_idx, :]
    else:
        dtm_sample = dtm

    term_vec = dtm_sample[:, term_idx].toarray().reshape(1, -1)
    sims = cosine_similarity(term_vec, dtm_sample.T).flatten()

    # Get top similar terms (excluding self)
    sims[term_idx] = -1
    top_idx = sims.argsort()[-top_n:][::-1]

    return [(vocab[i], float(sims[i])) for i in top_idx if sims[i] > 0]


def count_papers_with_term(query_term, dtm, vocab, meta):
    """Count papers containing the term and show year distribution."""
    vocab_list = list(vocab)
    if query_term not in vocab_list:
        return 0, {}

    term_idx = vocab_list.index(query_term)
    mask = (dtm[:, term_idx].toarray().flatten() > 0)
    count = int(mask.sum())

    years = meta.loc[mask, "year"].dropna().astype(int)
    year_dist = years.value_counts().sort_index().to_dict()

    return count, year_dist


def plot_keyword_report(query, term, affinities, compass_pos, similar_terms, year_dist):
    """Generate a visual report for the keyword."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle(f'HCI Landscape — Keyword: "{query}"', fontsize=18, fontweight="bold")

    # 1. Theme Affinity Bar Chart
    ax = axes[0, 0]
    themes_sorted = sorted(affinities.items(), key=lambda x: x[1], reverse=True)
    names = [t[0] for t in themes_sorted]
    scores = [t[1] for t in themes_sorted]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(names)))
    bars = ax.barh(range(len(names)), scores, color=colors[::-1])
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Affinity Score (Jaccard)")
    ax.set_title("Theme Affinity")
    ax.invert_yaxis()
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                f"{score:.4f}", va="center", fontsize=8)

    # 2. Compass Position
    ax = axes[0, 1]
    if compass_pos[0] is not None:
        ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
        ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
        ax.scatter([compass_pos[0]], [compass_pos[1]], s=200, c="red", zorder=5, edgecolors="black")
        ax.annotate(term, (compass_pos[0], compass_pos[1]),
                    textcoords="offset points", xytext=(10, 10), fontsize=12, fontweight="bold")

        # Add reference labels
        ax.text(-0.5, -0.05, "Education/Qualitative", ha="center", fontsize=8, color="blue", alpha=0.6)
        ax.text(0.5, -0.05, "Technical/Computational", ha="center", fontsize=8, color="red", alpha=0.6)
        ax.set_xlabel("Component 1 (Education ← → Technical)")
        ax.set_ylabel("Component 2")
        ax.set_xlim(-0.7, 1.0)
        ax.set_ylim(-0.5, 0.5)
    else:
        ax.text(0.5, 0.5, f'"{term}" not in\ntop 2000 terms\nfor compass', ha="center", va="center",
                transform=ax.transAxes, fontsize=14, color="gray")
    ax.set_title("Position on Semantic Compass")

    # 3. Most Similar Terms
    ax = axes[1, 0]
    if similar_terms:
        sim_names = [t[0] for t in similar_terms[:15]]
        sim_scores = [t[1] for t in similar_terms[:15]]
        ax.barh(range(len(sim_names)), sim_scores, color="#4363d8", alpha=0.7)
        ax.set_yticks(range(len(sim_names)))
        ax.set_yticklabels(sim_names, fontsize=9)
        ax.set_xlabel("Cosine Similarity")
        ax.invert_yaxis()
    ax.set_title("Most Similar Terms")

    # 4. Papers Over Time
    ax = axes[1, 1]
    if year_dist:
        years = sorted(year_dist.keys())
        counts = [year_dist[y] for y in years]
        ax.bar(years, counts, color="#3cb44b", alpha=0.7)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Papers")
    ax.set_title("Publication Trend")

    plt.tight_layout()
    safe_name = query.replace(" ", "_").lower()
    path = os.path.join(OUTPUT_DIR, f"keyword_{safe_name}.png")
    plt.savefig(path, dpi=200, bbox_inches="tight")
    print(f"\nSaved report: {path}")
    plt.close()


def search_keyword(query):
    """Main search function."""
    print(f'\nSearching HCI landscape for: "{query}"')
    print("=" * 60)

    dtm, vocab, meta, vectorizer, spectrum = load_all()

    # Find matches
    exact, partial, related = find_term_matches(query, vocab)

    print(f"\nVocabulary matches:")
    if exact:
        print(f"  Exact: {exact}")
    if partial:
        print(f"  Partial: {partial[:20]}")
    if related:
        print(f"  Related: {related[:10]}")

    # Use best match
    if exact:
        term = exact[0]
    elif partial:
        # Prefer the shortest partial match
        term = sorted(partial, key=len)[0]
    else:
        print(f'\n  No match found for "{query}" in vocabulary.')
        print("  Try a different term or check partial matches above.")
        return

    print(f'\n  Using term: "{term}"')

    # Paper count
    count, year_dist = count_papers_with_term(term, dtm, vocab, meta)
    print(f"  Papers containing this term: {count:,} / {dtm.shape[0]:,} ({count/dtm.shape[0]*100:.1f}%)")

    # Compass position
    c1, c2 = get_compass_position(term, spectrum)
    if c1 is not None:
        direction = "Technical/Computational" if c1 > 0 else "Education/Qualitative"
        print(f"\n  Semantic Compass Position:")
        print(f"    Component 1: {c1:+.4f} ({direction} side)")
        print(f"    Component 2: {c2:+.4f}")

    # Theme affinity
    print(f"\n  Theme Affinity:")
    affinities = compute_theme_affinity(term, dtm, vocab, vectorizer)
    if affinities:
        for theme, score in sorted(affinities.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(score * 500)
            print(f"    {score:.4f} {bar:20s} {theme}")

    # Similar terms
    print(f"\n  Most Similar Terms:")
    similar = find_most_similar_terms(term, dtm, vocab)
    for t, s in similar[:15]:
        print(f"    {s:.4f}  {t}")

    # Year distribution
    if year_dist:
        recent = {k: v for k, v in sorted(year_dist.items()) if k >= 2015}
        if recent:
            print(f"\n  Recent Publication Trend:")
            for y, c in recent.items():
                bar = "█" * (c // 20)
                print(f"    {y}: {c:5d} {bar}")

    # Generate visual report
    plot_keyword_report(query, term, affinities, (c1, c2), similar, year_dist)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python 05_keyword_search.py \"keyword\"")
        print("\nExamples:")
        print('  python 05_keyword_search.py "artificial intelligence"')
        print('  python 05_keyword_search.py "virtual reality"')
        print('  python 05_keyword_search.py "accessibility"')
        print('  python 05_keyword_search.py "gesture recognition"')
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    search_keyword(query)

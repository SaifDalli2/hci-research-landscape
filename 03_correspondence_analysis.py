"""
Step 3: Correspondence Analysis (Map 1 — The Semantic Compass)

Compresses the term co-occurrence information into a low-dimensional spectrum.
This reveals the primary axis of variation across the HCI vocabulary —
e.g., design/qualitative vs systems/technical.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import prince
from scipy import sparse

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "processed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")


def load_data():
    dtm = sparse.load_npz(os.path.join(DATA_DIR, "dtm.npz"))
    vocab = np.load(os.path.join(DATA_DIR, "vocab.npy"), allow_pickle=True)
    meta = pd.read_csv(os.path.join(DATA_DIR, "metadata.csv"))
    return dtm, vocab, meta


from stopwords import is_valid_term


def build_cooccurrence_matrix(dtm, vocab, top_n=2000):
    """
    Build a term co-occurrence matrix from the DTM.
    We select the top_n most frequent terms and compute their co-occurrence.
    Filters out non-English terms.
    """
    # Get term frequencies
    term_freq = np.asarray(dtm.sum(axis=0)).flatten()

    # Filter to English terms only
    valid_mask = np.array([is_valid_term(v) for v in vocab])
    filtered_freq = term_freq.copy()
    filtered_freq[~valid_mask] = 0

    top_indices = filtered_freq.argsort()[-top_n:][::-1]
    top_indices = top_indices[filtered_freq[top_indices] > 0]

    # Subset DTM to top terms
    dtm_sub = dtm[:, top_indices]
    top_vocab = vocab[top_indices]

    # Binary presence matrix
    binary = (dtm_sub > 0).astype(int)

    # Co-occurrence = binary^T @ binary
    cooc = (binary.T @ binary).toarray()
    np.fill_diagonal(cooc, 0)

    cooc_df = pd.DataFrame(cooc, index=top_vocab, columns=top_vocab)

    print(f"Co-occurrence matrix: {cooc_df.shape}")
    return cooc_df, top_vocab


def run_correspondence_analysis(cooc_df, n_components=5):
    """Run Correspondence Analysis on the co-occurrence matrix."""
    print("Running Correspondence Analysis...")

    # prince CA expects a contingency table (non-negative integers)
    ca = prince.CA(n_components=n_components, n_iter=10, random_state=42)
    ca = ca.fit(cooc_df)

    # Get row (=column since symmetric) coordinates
    coords = ca.row_coordinates(cooc_df)

    # Explained inertia
    inertia = ca.percentage_of_variance_
    print(f"Explained inertia per component: {[f'{v:.2f}%' for v in inertia]}")

    return ca, coords, inertia


def plot_semantic_compass(coords, vocab, inertia, top_n_labels=80):
    """Plot the 1D semantic compass (Component 1) and 2D scatter."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Map 1A: 1D Spectrum (Compass) ---
    fig, ax = plt.subplots(figsize=(20, 6))

    x = coords.iloc[:, 0].values
    y = np.zeros_like(x)

    ax.scatter(x, y, alpha=0.3, s=10, c=x, cmap="RdYlBu_r")

    # Label extreme terms
    sorted_idx = np.argsort(x)
    label_indices = list(sorted_idx[:top_n_labels // 2]) + list(sorted_idx[-top_n_labels // 2:])

    for i in label_indices:
        ax.annotate(
            vocab[i], (x[i], 0),
            textcoords="offset points",
            xytext=(0, np.random.randint(-30, 30)),
            fontsize=6, alpha=0.7,
            arrowprops=dict(arrowstyle="-", alpha=0.3, lw=0.5),
        )

    ax.set_xlabel(f"Component 1 ({inertia[0]:.1f}% inertia)")
    ax.set_yticks([])
    ax.set_title("HCI Semantic Compass — Primary Axis of Variation")
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "map1_semantic_compass.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "map1_semantic_compass.pdf"), bbox_inches="tight")
    print(f"Saved: map1_semantic_compass.png/pdf")
    plt.close()

    # --- Map 1B: 2D Scatter (Components 1 & 2) ---
    fig, ax = plt.subplots(figsize=(16, 12))

    x = coords.iloc[:, 0].values
    y_vals = coords.iloc[:, 1].values

    ax.scatter(x, y_vals, alpha=0.3, s=10, c=x, cmap="RdYlBu_r")

    # Label most extreme points
    distances = np.sqrt(x**2 + y_vals**2)
    extreme_idx = distances.argsort()[-100:]

    for i in extreme_idx:
        ax.annotate(
            vocab[i], (x[i], y_vals[i]),
            fontsize=5, alpha=0.7,
        )

    ax.set_xlabel(f"Component 1 ({inertia[0]:.1f}% inertia)")
    ax.set_ylabel(f"Component 2 ({inertia[1]:.1f}% inertia)")
    ax.set_title("HCI Vocabulary — Correspondence Analysis (2D)")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "map1_ca_2d.png"), dpi=300, bbox_inches="tight")
    plt.savefig(os.path.join(OUTPUT_DIR, "map1_ca_2d.pdf"), bbox_inches="tight")
    print(f"Saved: map1_ca_2d.png/pdf")
    plt.close()


def save_spectrum_data(coords, vocab):
    """Save the spectrum data for further analysis."""
    spectrum_df = pd.DataFrame({
        "term": vocab,
        "component_1": coords.iloc[:, 0].values,
        "component_2": coords.iloc[:, 1].values,
    })
    spectrum_df = spectrum_df.sort_values("component_1")
    spectrum_df.to_csv(os.path.join(OUTPUT_DIR, "semantic_spectrum.csv"), index=False)

    print("\nSemantic Compass — Extremes:")
    print("LEFT end (negative):")
    for _, row in spectrum_df.head(20).iterrows():
        print(f"  {row['component_1']:+.4f}  {row['term']}")
    print("RIGHT end (positive):")
    for _, row in spectrum_df.tail(20).iterrows():
        print(f"  {row['component_1']:+.4f}  {row['term']}")


def main():
    dtm, vocab, meta = load_data()
    cooc_df, top_vocab = build_cooccurrence_matrix(dtm, vocab, top_n=2000)
    ca, coords, inertia = run_correspondence_analysis(cooc_df)
    plot_semantic_compass(coords, top_vocab, inertia)
    save_spectrum_data(coords, top_vocab)
    print("\nDone! Check the output/ directory for results.")


if __name__ == "__main__":
    main()

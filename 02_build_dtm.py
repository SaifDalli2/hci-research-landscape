"""
Step 2: Build Document-Term Matrix (DTM) from collected abstracts.

- Loads paper abstracts from JSONL (samples if dataset is very large)
- Tokenizes and cleans text
- Removes stopwords and low-frequency terms
- Builds a sparse DTM using TF-IDF weighting
- Saves the DTM and vocabulary for downstream analysis
"""

import json
import os
import pickle
import random
import re

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
INPUT_FILE = os.path.join(DATA_DIR, "hci_papers_filtered.jsonl")
OUTPUT_DIR = os.path.join(DATA_DIR, "processed")

# Max papers to use for DTM (memory constraint). 2M is 2x the neuroscience article.
MAX_PAPERS = 2_000_000


def load_papers(path, max_papers=None):
    """Load papers from JSONL file, optionally sampling."""
    # First pass: count lines
    print("Counting papers...")
    total = 0
    with open(path, "r") as f:
        for _ in f:
            total += 1
    print(f"Total papers in file: {total:,}")

    # Determine sampling
    if max_papers and total > max_papers:
        sample_rate = max_papers / total
        print(f"Sampling ~{max_papers:,} papers ({sample_rate:.1%} of {total:,})")
        random.seed(42)
        use_sampling = True
    else:
        use_sampling = False

    papers = []
    with open(path, "r") as f:
        for line in tqdm(f, total=total, desc="Loading papers"):
            if use_sampling and random.random() > sample_rate:
                continue
            paper = json.loads(line.strip())
            if paper.get("abstract"):
                papers.append(paper)
            if max_papers and len(papers) >= max_papers:
                break

    print(f"Loaded {len(papers):,} papers with abstracts")
    return papers


def clean_text(text):
    """Basic text cleaning."""
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_dtm(papers, min_df=10, max_df=0.5, max_features=20000):
    """
    Build TF-IDF Document-Term Matrix.

    Args:
        papers: list of paper dicts
        min_df: minimum document frequency (absolute count)
        max_df: maximum document frequency (proportion)
        max_features: max vocabulary size
    """
    print("Cleaning abstracts...")
    abstracts = [clean_text(p["abstract"]) for p in tqdm(papers, desc="Cleaning")]

    print(f"\nBuilding DTM from {len(abstracts):,} abstracts...")
    print(f"  min_df={min_df}, max_df={max_df}, max_features={max_features}")

    vectorizer = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),  # unigrams and bigrams
        sublinear_tf=True,   # apply log normalization
    )

    dtm = vectorizer.fit_transform(abstracts)
    vocab = vectorizer.get_feature_names_out()

    print(f"  DTM shape: {dtm.shape[0]:,} x {dtm.shape[1]:,} (papers x terms)")
    print(f"  Vocabulary size: {len(vocab):,}")
    print(f"  Sparsity: {1 - dtm.nnz / (dtm.shape[0] * dtm.shape[1]):.4%}")

    return dtm, vocab, vectorizer


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    papers = load_papers(INPUT_FILE, max_papers=MAX_PAPERS)
    dtm, vocab, vectorizer = build_dtm(papers)

    # Save metadata as DataFrame
    print("Building metadata...")
    meta_df = pd.DataFrame([
        {
            "id": p.get("id", ""),
            "title": p.get("title") or "",
            "year": p.get("year"),
            "cited_by_count": p.get("cited_by_count", 0),
            "venue": p.get("venue") or "",
            "num_authors": p.get("num_authors", 0),
        }
        for p in papers
    ])

    # Save everything
    print("Saving...")
    sparse.save_npz(os.path.join(OUTPUT_DIR, "dtm.npz"), dtm)
    np.save(os.path.join(OUTPUT_DIR, "vocab.npy"), vocab)
    meta_df.to_csv(os.path.join(OUTPUT_DIR, "metadata.csv"), index=False)
    with open(os.path.join(OUTPUT_DIR, "vectorizer.pkl"), "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"\nSaved to {OUTPUT_DIR}:")
    print(f"  dtm.npz — sparse DTM matrix ({dtm.shape[0]:,} x {dtm.shape[1]:,})")
    print(f"  vocab.npy — vocabulary array ({len(vocab):,} terms)")
    print(f"  metadata.csv — paper metadata ({len(meta_df):,} rows)")
    print(f"  vectorizer.pkl — fitted TfidfVectorizer")


if __name__ == "__main__":
    main()

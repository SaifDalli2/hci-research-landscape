"""
Parallel paper collector — runs multiple queries concurrently using separate
output files and checkpoints. Each worker respects rate limits independently.

Workers write to data/hci_papers_wN.jsonl with checkpoints in data/checkpoint_wN.json.
After all workers finish, run merge step to combine into hci_papers.jsonl.
"""

import requests
import json
import time
import os
import sys
import multiprocessing
from pathlib import Path

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
MAIN_FILE = os.path.join(DATA_DIR, "hci_papers.jsonl")
BASE_URL = "https://api.openalex.org/works"
EMAIL = "research@example.com"

# All 20 queries from the original script
ALL_QUERIES = [
    "human-computer interaction",
    "user interface design",
    "user experience",
    "usability study",
    "accessibility technology",
    "virtual reality interaction",
    "augmented reality interface",
    "brain-computer interface",
    "tangible interaction",
    "gesture recognition interface",
    "gaze tracking interaction",
    "voice user interface",
    "haptic feedback interaction",
    "computer-supported cooperative work",
    "interactive systems design",
    "information visualization interaction",
    "mobile interaction design",
    "wearable computing interaction",
    "ubiquitous computing",
    "natural user interface",
]

# Queries 0-3 are handled by the original process.
# Split remaining 16 queries (indices 4-19) across 3 workers.
WORKER_QUERIES = {
    1: [4, 5, 6, 7, 8],       # accessibility, VR, AR, BCI, tangible
    2: [9, 10, 11, 12, 13],    # gesture, gaze, voice, haptic, CSCW
    3: [14, 15, 16, 17, 18, 19],  # interactive sys, infovis, mobile, wearable, ubicomp, NUI
}

# Rate limit: each worker waits this long between requests
# 3 workers * 1.0s = ~3 req/s total, well under OpenAlex's 10 req/s limit
DELAY_PER_REQUEST = 1.0


def reconstruct_abstract(inverted_index):
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def extract_venue(work):
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    return source.get("display_name", "")


def load_seen_ids():
    """Load all existing paper IDs from the main file to skip duplicates."""
    seen = set()
    if os.path.exists(MAIN_FILE):
        print(f"  Loading existing IDs from main file...")
        with open(MAIN_FILE, "r") as f:
            for line in f:
                try:
                    seen.add(json.loads(line.strip())["id"])
                except:
                    pass
    print(f"  {len(seen)} existing IDs loaded")
    return seen


def load_worker_checkpoint(worker_id):
    cp_file = os.path.join(DATA_DIR, f"checkpoint_w{worker_id}.json")
    if os.path.exists(cp_file):
        with open(cp_file, "r") as f:
            return json.load(f)
    return None


def save_worker_checkpoint(worker_id, query_idx_in_list, cursor, page_count):
    cp_file = os.path.join(DATA_DIR, f"checkpoint_w{worker_id}.json")
    with open(cp_file, "w") as f:
        json.dump({
            "query_idx_in_list": query_idx_in_list,
            "cursor": cursor,
            "page_count": page_count,
        }, f)


def clear_worker_checkpoint(worker_id):
    cp_file = os.path.join(DATA_DIR, f"checkpoint_w{worker_id}.json")
    if os.path.exists(cp_file):
        os.remove(cp_file)


def fetch_query_worker(query, query_global_idx, worker_id, query_idx_in_list,
                       seen_ids, output_handle, start_cursor="*", start_page=0):
    """Fetch all papers for one query. Returns (new_count, completed)."""
    cursor = start_cursor
    page_count = start_page
    new_count = 0
    dup_count = 0
    backoff_time = 5
    max_consecutive_errors = 50
    consecutive_errors = 0

    tag = f"[W{worker_id}]"

    if start_page > 0:
        print(f'{tag} Resuming: "{query}" from page {start_page}')
    else:
        print(f'{tag} Starting: "{query}"')

    while cursor:
        params = {
            "search": query,
            "filter": "type:article",
            "per_page": 200,
            "cursor": cursor,
            "mailto": EMAIL,
        }

        try:
            response = requests.get(BASE_URL, params=params, timeout=30)
            if response.status_code == 429:
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    save_worker_checkpoint(worker_id, query_idx_in_list, cursor, page_count)
                    print(f"{tag} Too many rate limits. Checkpointing and stopping.", flush=True)
                    return new_count, False
                wait = min(backoff_time, 600)
                print(f"{tag} Rate limited. Waiting {wait}s...")
                time.sleep(wait)
                backoff_time *= 2
                continue
            response.raise_for_status()
            data = response.json()
            backoff_time = 5
            consecutive_errors = 0
        except requests.exceptions.RequestException as e:
            consecutive_errors += 1
            if consecutive_errors >= max_consecutive_errors:
                save_worker_checkpoint(worker_id, query_idx_in_list, cursor, page_count)
                print(f"{tag} Too many errors. Checkpointing and stopping.", flush=True)
                return new_count, False
            wait = min(backoff_time, 600)
            err_short = str(e)[:80]
            print(f"{tag} Error: {err_short}. Retry in {wait}s...")
            time.sleep(wait)
            backoff_time *= 2
            continue

        results = data.get("results", [])
        if not results:
            break

        for work in results:
            work_id = work.get("id", "")
            if work_id in seen_ids:
                dup_count += 1
                continue

            abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
            if not abstract or len(abstract) < 50:
                continue

            paper = {
                "id": work_id,
                "doi": work.get("doi", ""),
                "title": work.get("title", ""),
                "year": work.get("publication_year"),
                "abstract": abstract,
                "cited_by_count": work.get("cited_by_count", 0),
                "topics": [
                    {"id": t.get("id", ""), "name": t.get("display_name", ""), "score": t.get("score", 0)}
                    for t in (work.get("topics") or [])
                ],
                "venue": extract_venue(work),
                "num_authors": len(work.get("authorships") or []),
            }

            seen_ids.add(work_id)
            output_handle.write(json.dumps(paper) + "\n")
            new_count += 1

        page_count += 1
        cursor = data.get("meta", {}).get("next_cursor")
        total = data.get("meta", {}).get("count", "?")

        if page_count % 25 == 0:
            output_handle.flush()
            save_worker_checkpoint(worker_id, query_idx_in_list, cursor, page_count)
            print(f"{tag} Pages: {page_count} | New: {new_count} | Dups: {dup_count} | Available: {total}", flush=True)

        time.sleep(DELAY_PER_REQUEST)

    print(f"{tag} Done \"{query}\": +{new_count} new ({dup_count} dups)")
    return new_count, True


def run_worker(worker_id):
    """Run a single worker that processes its assigned queries sequentially."""
    query_indices = WORKER_QUERIES[worker_id]
    output_file = os.path.join(DATA_DIR, f"hci_papers_w{worker_id}.jsonl")
    tag = f"[W{worker_id}]"

    print(f"{tag} Starting. Queries: {[ALL_QUERIES[i] for i in query_indices]}", flush=True)

    # Only load IDs from this worker's own file (lightweight).
    # Cross-dedup against main file happens at merge time.
    seen_ids = set()
    if os.path.exists(output_file):
        print(f"{tag} Loading IDs from own worker file...", flush=True)
        with open(output_file, "r") as f:
            for line in f:
                try:
                    seen_ids.add(json.loads(line.strip())["id"])
                except:
                    pass
        print(f"{tag} Loaded {len(seen_ids)} existing IDs from worker file", flush=True)

    # Check for checkpoint
    checkpoint = load_worker_checkpoint(worker_id)
    start_list_idx = 0
    start_cursor = "*"
    start_page = 0

    if checkpoint:
        start_list_idx = checkpoint["query_idx_in_list"]
        start_cursor = checkpoint["cursor"]
        start_page = checkpoint["page_count"]
        print(f"{tag} Resuming from query list index {start_list_idx}, page {start_page}")

    total_new = 0

    with open(output_file, "a") as f:
        for list_idx in range(start_list_idx, len(query_indices)):
            global_idx = query_indices[list_idx]
            query = ALL_QUERIES[global_idx]

            if list_idx == start_list_idx and start_cursor != "*":
                save_worker_checkpoint(worker_id, list_idx, start_cursor, start_page)
                new, completed = fetch_query_worker(
                    query, global_idx, worker_id, list_idx, seen_ids, f,
                    start_cursor=start_cursor, start_page=start_page,
                )
            else:
                save_worker_checkpoint(worker_id, list_idx, "*", 0)
                new, completed = fetch_query_worker(
                    query, global_idx, worker_id, list_idx, seen_ids, f,
                )

            total_new += new

            if not completed:
                save_worker_checkpoint(worker_id, list_idx, "*", 0)
                print(f"{tag} Stopped. Will resume on next run.")
                return False

            # Query completed — advance checkpoint
            if list_idx + 1 < len(query_indices):
                save_worker_checkpoint(worker_id, list_idx + 1, "*", 0)

    # All queries done
    clear_worker_checkpoint(worker_id)
    print(f"\n{tag} ALL QUERIES COMPLETE. Total new: {total_new}")
    return True


def run_single_worker(worker_id):
    """Entry point for a single worker (for use with auto-resume wrapper)."""
    print(f"\n{'='*60}")
    print(f"Parallel Collector — Worker {worker_id}")
    print(f"{'='*60}")
    success = run_worker(worker_id)
    if success:
        print(f"[W{worker_id}] Finished all assigned queries!")
    else:
        print(f"[W{worker_id}] Stopped (will resume on next run).")


def merge_worker_files():
    """Merge all worker output files into the main file, deduplicating."""
    print(f"\n{'='*60}")
    print("Merging worker files into main file...")
    print(f"{'='*60}")

    # Load existing IDs from main file
    seen_ids = set()
    if os.path.exists(MAIN_FILE):
        with open(MAIN_FILE, "r") as f:
            for line in f:
                try:
                    seen_ids.add(json.loads(line.strip())["id"])
                except:
                    pass
    print(f"  Main file has {len(seen_ids)} papers")

    total_added = 0
    with open(MAIN_FILE, "a") as out:
        for wid in sorted(WORKER_QUERIES.keys()):
            wfile = os.path.join(DATA_DIR, f"hci_papers_w{wid}.jsonl")
            if not os.path.exists(wfile):
                print(f"  Worker {wid}: no file found, skipping")
                continue
            added = 0
            with open(wfile, "r") as f:
                for line in f:
                    try:
                        paper = json.loads(line.strip())
                        if paper["id"] not in seen_ids:
                            out.write(line)
                            seen_ids.add(paper["id"])
                            added += 1
                    except:
                        pass
            print(f"  Worker {wid}: merged {added} new papers")
            total_added += added

    print(f"\nTotal merged: {total_added} new papers")
    print(f"Main file now has: {len(seen_ids)} unique papers")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python 01_collect_parallel.py worker <id>   — run worker 1, 2, or 3")
        print("  python 01_collect_parallel.py merge         — merge worker files into main")
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd == "worker" and len(sys.argv) >= 3:
        wid = int(sys.argv[2])
        if wid not in WORKER_QUERIES:
            print(f"Invalid worker ID {wid}. Must be 1, 2, or 3.")
            sys.exit(1)
        run_single_worker(wid)

    elif cmd == "merge":
        merge_worker_files()

    else:
        print("Unknown command. Use 'worker <id>' or 'merge'.")
        sys.exit(1)

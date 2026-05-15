"""
Step 1: Collect HCI paper abstracts from OpenAlex API.

Uses keyword search + topic filters to pull as many HCI papers as possible.
Streams results to disk incrementally to avoid memory issues.

Features:
- Cursor checkpointing: saves progress per query so restarts resume
  from the exact page, not from the beginning.
- Deduplication: skips papers already collected.
- Exponential backoff on rate limits.
"""

import requests
import json
import time
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_FILE = os.path.join(DATA_DIR, "hci_papers.jsonl")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "checkpoint.json")

BASE_URL = "https://api.openalex.org/works"
EMAIL = "research@example.com"

# Multiple search queries to cover HCI broadly
SEARCH_QUERIES = [
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


def load_checkpoint():
    """Load checkpoint: which query index and cursor we were at."""
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, "r") as f:
            return json.load(f)
    return {"query_index": 0, "cursor": "*", "page_count": 0}


def save_checkpoint(query_index, cursor, page_count):
    """Save current progress so we can resume exactly here."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({
            "query_index": query_index,
            "cursor": cursor,
            "page_count": page_count,
        }, f)


def clear_checkpoint():
    """Remove checkpoint file when a query completes."""
    if os.path.exists(CHECKPOINT_FILE):
        os.remove(CHECKPOINT_FILE)


def reconstruct_abstract(inverted_index):
    """Reconstruct abstract text from OpenAlex inverted index format."""
    if not inverted_index:
        return ""
    word_positions = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(word for _, word in word_positions)


def extract_venue(work):
    """Extract venue/journal name from work."""
    loc = work.get("primary_location") or {}
    source = loc.get("source") or {}
    return source.get("display_name", "")


def fetch_query(query, query_index, seen_ids, output_handle, start_cursor="*", start_page=0):
    """Fetch all papers for a search query using cursor pagination with checkpointing."""
    cursor = start_cursor
    page_count = start_page
    new_count = 0
    dup_count = 0
    backoff_time = 5
    max_consecutive_rate_limits = 50
    consecutive_rate_limits = 0

    if start_page > 0:
        print(f'\n  Resuming: "{query}" from page {start_page}')
    else:
        print(f'\n  Searching: "{query}"')

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
                consecutive_rate_limits += 1
                if consecutive_rate_limits >= max_consecutive_rate_limits:
                    print(f"    Hit {max_consecutive_rate_limits} consecutive rate limits. Saving checkpoint and stopping.")
                    save_checkpoint(query_index, cursor, page_count)
                    return new_count, False  # signal to stop
                wait = min(backoff_time, 600)
                print(f"    Rate limited ({consecutive_rate_limits}/{max_consecutive_rate_limits}). Waiting {wait}s...")
                time.sleep(wait)
                backoff_time *= 2
                continue
            response.raise_for_status()
            data = response.json()
            backoff_time = 5
            consecutive_rate_limits = 0
        except requests.exceptions.RequestException as e:
            consecutive_rate_limits += 1
            if consecutive_rate_limits >= max_consecutive_rate_limits:
                print(f"    Hit {max_consecutive_rate_limits} consecutive errors. Saving checkpoint and stopping.")
                save_checkpoint(query_index, cursor, page_count)
                return new_count, False
            wait = min(backoff_time, 600)
            print(f"    Error: {e}. Retrying in {wait}s...")
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
            save_checkpoint(query_index, cursor, page_count)
            print(f"    Pages: {page_count} | New: {new_count} | Dups: {dup_count} | Available: {total}")

        # Respect rate limits — 0.5s between requests to avoid 429s
        time.sleep(0.5)

    print(f"    Done: +{new_count} new papers ({dup_count} duplicates skipped)")
    return new_count, True  # True = completed successfully


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    seen_ids = set()

    # Resume support: load existing IDs if file exists
    if os.path.exists(OUTPUT_FILE) and os.path.getsize(OUTPUT_FILE) > 0:
        print("Loading existing papers...")
        with open(OUTPUT_FILE, "r") as f:
            for line in f:
                try:
                    paper = json.loads(line.strip())
                    seen_ids.add(paper["id"])
                except:
                    pass
        print(f"  Found {len(seen_ids)} existing papers")

    # Load checkpoint
    checkpoint = load_checkpoint()
    start_query_index = checkpoint["query_index"]
    start_cursor = checkpoint["cursor"]
    start_page = checkpoint["page_count"]

    if start_query_index > 0 or start_cursor != "*":
        print(f"  Resuming from query {start_query_index} (\"{SEARCH_QUERIES[start_query_index]}\"), page {start_page}")

    total_new = 0

    with open(OUTPUT_FILE, "a") as f:
        for i, query in enumerate(SEARCH_QUERIES):
            # Skip completed queries
            if i < start_query_index:
                continue

            # For the checkpoint query, resume from saved cursor
            if i == start_query_index and start_cursor != "*":
                new, completed = fetch_query(query, i, seen_ids, f,
                                             start_cursor=start_cursor,
                                             start_page=start_page)
            else:
                new, completed = fetch_query(query, i, seen_ids, f)

            total_new += new
            print(f"  Running total: {len(seen_ids)} unique papers")

            if not completed:
                print(f"\n  Stopped due to rate limiting. Run again to resume from here.")
                break

            # Query completed — clear checkpoint and move to next
            clear_checkpoint()
            # Save that we're starting the next query
            if i + 1 < len(SEARCH_QUERIES):
                save_checkpoint(i + 1, "*", 0)
        else:
            # All queries completed
            clear_checkpoint()
            print(f"\n  All queries completed!")

    print(f"\n{'='*60}")
    print(f"Total unique papers: {len(seen_ids)}")
    print(f"New papers this run: {total_new}")
    print(f"Saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()

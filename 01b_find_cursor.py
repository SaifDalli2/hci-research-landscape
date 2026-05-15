"""
Fast-forward through pages to find the cursor at page 1225.
Only fetches minimal data (1 result per page) to go fast.
Saves the cursor to checkpoint.json when done.
"""

import requests
import json
import time
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
CHECKPOINT_FILE = os.path.join(DATA_DIR, "checkpoint.json")
BASE_URL = "https://api.openalex.org/works"
EMAIL = "research@example.com"

TARGET_PAGE = 1225
QUERY = "human-computer interaction"

cursor = "*"
page = 0
backoff = 5

print(f"Fast-forwarding to page {TARGET_PAGE} for query: {QUERY}")
print("Using per_page=200 to match original pagination...")

while cursor and page < TARGET_PAGE:
    params = {
        "search": QUERY,
        "filter": "type:article",
        "per_page": 200,
        "cursor": cursor,
        "select": "id",  # minimal data for speed
        "mailto": EMAIL,
    }

    try:
        r = requests.get(BASE_URL, params=params, timeout=30)
        if r.status_code == 429:
            wait = min(backoff, 120)
            retries = retries + 1 if 'retries' in dir() else 1
            if retries > 20:
                print(f"\n  Still rate-limited after {retries} retries. Try again later.")
                exit(1)
            print(f"  Rate limited ({retries}/20). Waiting {wait}s...")
            time.sleep(wait)
            backoff *= 2
            continue
        r.raise_for_status()
        data = r.json()
        backoff = 5
        retries = 0
    except Exception as e:
        print(f"  Error: {e}. Retrying...")
        time.sleep(5)
        continue

    cursor = data.get("meta", {}).get("next_cursor")
    page += 1

    if page % 100 == 0:
        print(f"  Page {page}/{TARGET_PAGE}")

    time.sleep(0.1)

if cursor:
    checkpoint = {"query_index": 0, "cursor": cursor, "page_count": TARGET_PAGE}
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f)
    print(f"\nDone! Saved checkpoint at page {TARGET_PAGE}")
    print(f"Cursor: {cursor[:50]}...")
else:
    print("Failed — ran out of results before reaching target page.")

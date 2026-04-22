from playwright.sync_api import sync_playwright
import csv
import time
import re
import os
from datetime import datetime

SEARCH_QUERY = "grok"           # The query where we want to scrape tweets from
SESSION_FILE = "session.json"
CSV_FILE     = "data/initial_tweets.csv"
MAX_SCROLLS  = 100              # Captures ~500-1000 tweets depending on feed activity
SCROLL_PAUSE = 2.5              # Seconds between scrolls
SCROLL_PX    = 2500             # Pixels per scroll

CSV_FIELDS = ["id", "text", "author", "handle", "timestamp",
              "likes", "retweets", "replies", "tweet_url", "scraped_at"]


def load_seen_ids(csv_path: str) -> set:
    #Read already-saved tweet IDs from existing CSV to avoid duplicates.
    seen = set()
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("id"):
                    seen.add(row["id"])
    return seen


def parse_count(raw: str) -> int:
    """Convert '1.2K', '3M', '' → int."""
    if not raw:
        return 0
    raw = raw.strip().replace(",", "")
    try:
        if raw.endswith("K"):
            return int(float(raw[:-1]) * 1_000)
        if raw.endswith("M"):
            return int(float(raw[:-1]) * 1_000_000)
        return int(raw)
    except ValueError:
        return 0


def extract_tweet_id(url: str) -> str | None:
    # Pull tweet ID from its URL.
    m = re.search(r"/status/(\d+)", url or "")
    return m.group(1) if m else None


def scrape_tweets_from_page(page) -> list[dict]:
    """Extract all visible tweet data from the current DOM."""
    return page.evaluate("""
        () => {
            const results = [];
            const articles = document.querySelectorAll('article[data-testid="tweet"]');

            articles.forEach(article => {
                try {
                    const textEl = article.querySelector('[data-testid="tweetText"]');
                    const text = textEl ? textEl.innerText : "";

                    const userEl = article.querySelector('[data-testid="User-Name"]');
                    const spans = userEl ? userEl.querySelectorAll('span') : [];
                    let author = "", handle = "";
                    spans.forEach(s => {
                        const t = s.innerText.trim();
                        if (t.startsWith("@")) handle = t;
                        else if (t && !author) author = t;
                    });

                    const timeEl = article.querySelector("time");
                    const timestamp = timeEl ? timeEl.getAttribute("datetime") : "";

                    const linkEl = article.querySelector('a[href*="/status/"]');
                    const tweetUrl = linkEl ? linkEl.href : "";

                    const replyEl = article.querySelector('[data-testid="reply"]   [data-testid="app-text-transition-container"]');
                    const rtEl    = article.querySelector('[data-testid="retweet"] [data-testid="app-text-transition-container"]');
                    const likeEl  = article.querySelector('[data-testid="like"]    [data-testid="app-text-transition-container"]');

                    results.push({
                        text, author, handle, timestamp,
                        tweet_url: tweetUrl,
                        replies:  replyEl ? replyEl.innerText.trim() : "0",
                        retweets: rtEl    ? rtEl.innerText.trim()    : "0",
                        likes:    likeEl  ? likeEl.innerText.trim()  : "0",
                    });
                } catch(e) {}
            });
            return results;
        }
    """)


def save_tweets(writer, seen_ids: set, tweets: list[dict]) -> int:

    new_count = 0
    for t in tweets:
        tweet_id = extract_tweet_id(t.get("tweet_url", ""))
        if not tweet_id or tweet_id in seen_ids:
            continue
        seen_ids.add(tweet_id)
        writer.writerow({
            "id":         tweet_id,
            "text":       t.get("text", ""),
            "author":     t.get("author", ""),
            "handle":     t.get("handle", ""),
            "timestamp":  t.get("timestamp", ""),
            "likes":      parse_count(t.get("likes", "0")),
            "retweets":   parse_count(t.get("retweets", "0")),
            "replies":    parse_count(t.get("replies", "0")),
            "tweet_url":  t.get("tweet_url", ""),
            "scraped_at": datetime.utcnow().isoformat(),
        })
        new_count += 1
    return new_count


def main():
    seen_ids = load_seen_ids(CSV_FILE)
    is_new_file = not os.path.exists(CSV_FILE)
    print(f"Output: {os.path.abspath(CSV_FILE)}")
    if seen_ids:
        print(f"   (Resuming — {len(seen_ids)} tweets already in file)")

    csv_file = open(CSV_FILE, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
    if is_new_file:
        writer.writeheader()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

        # ── Session handling ─────────────────────────────────────────────
        if os.path.exists(SESSION_FILE):
            print(f"🔑 Reusing saved session from {SESSION_FILE}")
            context = browser.new_context(storage_state=SESSION_FILE)
        else:
            context = browser.new_context()
            page = context.new_page()
            page.goto("https://x.com/login")
            input("Log in manually in the browser, then press Enter here...")
            context.storage_state(path=SESSION_FILE)
            print(f"Session saved to {SESSION_FILE}")

        page = context.new_page()


        search_url = f"https://x.com/search?q={SEARCH_QUERY}&f=live"
        print(f"🔍 Searching: {search_url}")
        page.goto(search_url)
        page.wait_for_selector('article[data-testid="tweet"]', timeout=15000)
        time.sleep(2)


        total_saved = 0
        no_new_streak = 0

        for scroll_num in range(1, MAX_SCROLLS + 1):
            raw = scrape_tweets_from_page(page)
            new = save_tweets(writer, seen_ids, raw)
            csv_file.flush()   
            total_saved += new

            print(f"  Scroll {scroll_num:>3}/{MAX_SCROLLS} | "
                  f"+{new:>3} new | {len(seen_ids):>4} total")

            if new == 0:
                no_new_streak += 1
                if no_new_streak >= 5:
                    print("⚠️  No new tweets for 5 scrolls — feed exhausted or rate limited.")
                    break
            else:
                no_new_streak = 0

            page.mouse.wheel(0, SCROLL_PX)
            time.sleep(SCROLL_PAUSE)

        browser.close()

    csv_file.close()
    print(f"Done! {len(seen_ids)} tweets saved to '{CSV_FILE}'")


if __name__ == "__main__":
    main()
    
## Had help from Claude for this scrapper.
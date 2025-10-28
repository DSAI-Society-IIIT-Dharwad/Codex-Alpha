import requests
from bs4 import BeautifulSoup
import csv
import time

BASE_URL = "https://www.kaanoon.com"
START_URL = f"{BASE_URL}/answers"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36"
}

def get_total_pages():
    """Find how many pages exist."""
    r = requests.get(START_URL, headers=HEADERS)
    s = BeautifulSoup(r.text, "html.parser")
    pages = s.select(".pagination li a")
    if not pages:
        return 1
    numbers = [int(a.text) for a in pages if a.text.isdigit()]
    return max(numbers) if numbers else 1


def scrape_page(page_number):
    """Scrape titles and links from a single page."""
    url = f"{START_URL}?page={page_number}"
    res = requests.get(url, headers=HEADERS)
    soup = BeautifulSoup(res.text, "html.parser")

    questions = soup.select("table.landing-latest-answers tr.answer")
    data = []

    for q in questions:
        title_tag = q.select_one("span.title a")
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)
        href = title_tag.get("href")
        full_link = BASE_URL + href if href else ""
        data.append([title, full_link])
        print(f"✅ Page {page_number}: {title[:80]}...")

    return data


def main():
    total_pages = get_total_pages()
    print(f"📄 Found {total_pages} pages to scrape.")

    with open("kaanoon_titles_links.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Title", "Full Link"])

        for page in range(1, 20000 + 1):
            page_data = scrape_page(page)
            writer.writerows(page_data)
            time.sleep(2)  # polite delay

    print("\n✅ Scraping completed! Data saved to 'kaanoon_titles_links.csv'.")


if __name__ == "__main__":
    main()

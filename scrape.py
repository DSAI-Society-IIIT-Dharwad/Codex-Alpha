import requests
from bs4 import BeautifulSoup
import csv
import time

# Base URL
BASE_URL = "https://www.kaanoon.com"

# Target page
URL = f"{BASE_URL}/answers"

# Headers to mimic a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}

# Fetch the main answers page
response = requests.get(URL, headers=HEADERS)
soup = BeautifulSoup(response.text, "html.parser")

# Find all question rows
questions = soup.select("table.landing-latest-answers tr.answer")

# Prepare CSV
with open("kaanoon_data.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Title", "Link", "Description"])  # CSV headers

    for q in questions:
        # Extract title
        title_tag = q.select_one("span.title a")
        if not title_tag:
            continue
        title = title_tag.get_text(strip=True)

        # Extract href and form full link
        href = title_tag.get("href")
        full_link = BASE_URL + href if href else ""

        print(f"Scraping: {title}")

        # Visit each question link to get the full description
        try:
            detail_page = requests.get(full_link, headers=HEADERS)
            detail_soup = BeautifulSoup(detail_page.text, "html.parser")

            # Extract full description (question text)
            desc_tag = detail_soup.select_one("div.thread div.question.entry pre.description")
            description = desc_tag.get_text(strip=True) if desc_tag else "No description found."

            # Write to CSV
            writer.writerow([title, full_link, description])

        except Exception as e:
            print(f"Error scraping {full_link}: {e}")
            writer.writerow([title, full_link, "Error fetching description"])

        # Be polite to server
        time.sleep(2)

print("✅ Scraping completed! Data saved to 'kaanoon_data.csv'")

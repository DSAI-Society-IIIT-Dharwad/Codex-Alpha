import json
import requests
from bs4 import BeautifulSoup

# Step 1: Load URL from the JSON file
with open("search_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Take the top-ranked URL
top_url = data[0]["url"]
print(f"Scraping from: {top_url}")

# Step 2: Fetch page content
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
response = requests.get(top_url, headers=headers)
soup = BeautifulSoup(response.text, "html.parser")

# Step 3: Extract all answers
answers = []
for answer_div in soup.find_all("div", class_="answer entry"):
    # Extract main text
    text_div = answer_div.find("div", class_="text")
    text = text_div.get_text(strip=True, separator="\n") if text_div else ""
    
    # Extract lawyer name and location (if available)
    lawyer_info = answer_div.find("div", class_="lawyer")
    lawyer_name = lawyer_info.get_text(strip=True) if lawyer_info else "Unknown"
    
    answers.append({
        "lawyer": lawyer_name,
        "text": text
    })

# Step 4: Save to text file
with open("kaanoon_answers.txt", "w", encoding="utf-8") as f:
    for i, ans in enumerate(answers, 1):
        f.write(f"Answer {i} by {ans['lawyer']}:\n{ans['text']}\n\n{'-'*80}\n\n")

print(f"✅ Scraped {len(answers)} answers and saved to 'kaanoon_answers.txt'")

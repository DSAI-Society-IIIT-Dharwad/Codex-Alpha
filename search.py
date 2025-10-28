import json
import requests
from bs4 import BeautifulSoup
import time

# Step 1: Load all URLs from JSON
with open("search_results.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Step 2: Prepare output file
output_file = "kaanoon_all_answers.txt"

with open(output_file, "w", encoding="utf-8") as f_out:
    for item in data[:10]:  # top 10 results
        rank = item.get("rank")
        title = item.get("title", "No Title")
        url = item.get("url")
        
        # Section header
        f_out.write(f"{'='*100}\n")
        f_out.write(f"🔹 Rank {rank}: {title}\n")
        f_out.write(f"🔗 URL: {url}\n")
        f_out.write(f"{'='*100}\n\n")

        # Fetch page
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")

        # Extract answers
        answers = []
        for answer_div in soup.find_all("div", class_="answer entry"):
            text_div = answer_div.find("div", class_="text")
            text = text_div.get_text(strip=True, separator="\n") if text_div else ""
            
            lawyer_info = answer_div.find("div", class_="lawyer")
            lawyer_name = lawyer_info.get_text(strip=True) if lawyer_info else "Unknown"
            
            answers.append({"lawyer": lawyer_name, "text": text})
        
        # Write answers
        if answers:
            for i, ans in enumerate(answers, 1):
                f_out.write(f"Answer {i} by {ans['lawyer']}:\n{ans['text']}\n\n{'-'*80}\n\n")
        else:
            f_out.write("⚠️ No answers found.\n\n")
        
        f_out.write("\n\n")
        print(f"✅ Done scraping Rank {rank}")

print(f"\n🎉 All top 10 results scraped and saved in '{output_file}'")

import html
import re
import json
import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(s: str) -> str:
    """Clean and normalize text."""
    if not isinstance(s, str):
        return ""
    s = html.unescape(str(s).strip())
    s = re.sub(r"\s+", " ", s)
    return s


def rank_titles_hybrid(titles, query, top_k=10):
    """Rank titles using hybrid TF-IDF + Fuzzy logic."""
    clean_titles = [normalize_text(t) for t in titles]
    query = normalize_text(query)

    # TF-IDF cosine similarity
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(clean_titles + [query])
    query_vec = tfidf_matrix[-1]
    title_vecs = tfidf_matrix[:-1]
    cos_scores = cosine_similarity(title_vecs, query_vec).ravel()

    # Fuzzy partial ratio (scaled 0–1)
    fuzzy_scores = np.array([fuzz.partial_ratio(query, t) / 100.0 for t in clean_titles])

    # Weighted hybrid
    hybrid = 0.7 * cos_scores + 0.3 * fuzzy_scores

    # Get top-k
    top_idx = np.argsort(-hybrid)[:top_k]
    return [(int(i), float(hybrid[i])) for i in top_idx]


def main():
    csv_path = "kaanoon_titles_links.csv"  # file in same folder
    df = pd.read_csv(csv_path)

    # Use fixed columns
    title_col = "Title"
    link_col = "Full Link"

    titles = df[title_col].astype(str).tolist()
    links = df[link_col].astype(str).tolist()

    # Ask user for question
    question = input("\nEnter your question: ").strip()
    if not question:
        print("No question entered. Exiting.")
        return

    ranked = rank_titles_hybrid(titles, question, top_k=10)

    results = []
    print("\nTop 10 matching titles:\n")
    for rank, (idx, score) in enumerate(ranked, start=1):
        title = titles[idx]
        url = links[idx]
        results.append({"rank": rank, "score": round(score, 4), "title": title, "url": url})
        print(f"{rank}. [{round(score,4)}] {title}\n   {url}\n")

    # Optional: Save results as JSON
    with open("search_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("Results saved to 'search_results.json'.")


if __name__ == "__main__":
    main()
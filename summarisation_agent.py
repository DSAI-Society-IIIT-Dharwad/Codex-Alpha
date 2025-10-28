#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
single_file_rag_groq.py
Run with:  python single_file_rag_groq.py

What it does (no external installs):
- Loads and chunks 'kaanoon_all_answers.txt' from the SAME folder
- Builds a lightweight TF-IDF index (pure Python stdlib)
- Prompts you for a question (input())
- Retrieves top-k relevant chunks
- Calls Groq's Chat Completions REST API (OpenAI-compatible) via urllib (stdlib)
- Prints retrieved chunks and a concise, cited answer

Requirements:
- Environment variable: GROQ_API_KEY=<your_key>
- File: kaanoon_all_answers.txt in the same folder
"""

import os
import re
import json
import math
import time
import sys
from collections import Counter, defaultdict
from urllib import request, error
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Config (edit if needed)
# ----------------------------
KB_PATH = "kaanoon_all_answers.txt"
MODEL = "llama-3.1-70b-versatile"   # e.g., "llama-3.1-70b-versatile", "mixtral-8x7b-32768"
TOP_K = 6                            # retrieved chunks
MAX_CHUNK_TOKENS = 1200
CHUNK_OVERLAP = 150
API_URL = "https://api.groq.com/openai/v1/chat/completions"
REQUEST_TIMEOUT = 120  # seconds

# Minimal English stopword set (expand as needed)
STOPWORDS = {
    "the","a","an","and","or","but","if","then","else","when","while","of","for","to","in","on","at","by","from",
    "with","without","as","is","are","was","were","be","been","being","that","this","these","those","it","its",
    "into","over","under","above","below","than","so","such","not","no","nor","do","does","did","doing","done",
    "can","could","should","would","may","might","must","shall","will","you","your","yours","we","our","ours",
    "they","their","theirs","he","him","his","she","her","hers","i","me","my","mine"
}

# ----------------------------
# Text & Chunking
# ----------------------------
def normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def tokenize(text: str):
    # Lowercase, keep word characters, split
    words = re.findall(r"[A-Za-z0-9_]+", text.lower())
    return [w for w in words if w not in STOPWORDS]

def chunk_text(text: str, max_tokens: int = MAX_CHUNK_TOKENS, overlap: int = CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        j = min(len(words), i + max_tokens)
        chunk = " ".join(words[i:j]).strip()
        if chunk:
            chunks.append(chunk)
        step = j - overlap
        i = step if step > i else j  # avoid infinite loop when short text
    return chunks

# ----------------------------
# TF-IDF Index (pure stdlib)
# ----------------------------
class TFIDFIndex:
    def __init__(self, chunks):
        """
        Build a simple TF-IDF index over chunks.
        """
        self.chunks = [normalize_ws(c) for c in chunks]
        self.tokenized = [tokenize(c) for c in self.chunks]
        self.N = len(self.chunks)
        # Document frequency
        df = Counter()
        for toks in self.tokenized:
            for term in set(toks):
                df[term] += 1
        self.df = df
        # IDF with smoothing
        self.idf = {t: math.log((self.N + 1) / (df_t + 1)) + 1.0 for t, df_t in df.items()}
        # Precompute tf-idf vectors as dicts
        self.doc_vecs = []
        for toks in self.tokenized:
            tf = Counter(toks)
            length_sq = 0.0
            vec = {}
            for t, f in tf.items():
                w = (f / len(toks)) * self.idf.get(t, 0.0)
                if w != 0.0:
                    vec[t] = w
                    length_sq += w * w
            norm = math.sqrt(length_sq) if length_sq > 0 else 1.0
            # Store normalized vector
            self.doc_vecs.append({t: w / norm for t, w in vec.items()})

    def _vec_for_query(self, query: str):
        toks = tokenize(query)
        if not toks:
            return {}
        tf = Counter(toks)
        length_sq = 0.0
        vec = {}
        for t, f in tf.items():
            w = (f / len(toks)) * self.idf.get(t, 0.0)
            if w != 0.0:
                vec[t] = w
                length_sq += w * w
        norm = math.sqrt(length_sq) if length_sq > 0 else 1.0
        return {t: w / norm for t, w in vec.items()}

    @staticmethod
    def _cosine_sparse(v1: dict, v2: dict):
        if not v1 or not v2:
            return 0.0
        # iterate over smaller dict
        if len(v1) > len(v2):
            v1, v2 = v2, v1
        s = 0.0
        for t, w in v1.items():
            w2 = v2.get(t)
            if w2 is not None:
                s += w * w2
        return s

    def retrieve(self, query: str, k: int = 6):
        qv = self._vec_for_query(query)
        sims = []
        for idx, dv in enumerate(self.doc_vecs):
            sims.append((idx, self._cosine_sparse(qv, dv)))
        sims.sort(key=lambda x: x[1], reverse=True)
        return sims[:k]

# ----------------------------
# Groq Chat Completions via urllib (no external deps)
# ----------------------------
def call_groq_chat(api_key: str, model: str, system_prompt: str, user_prompt: str, temperature=0.2, max_tokens=700):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": False
        # You can add "response_format" or "stop" if needed
    }
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(API_URL, data=data, method="POST")
    req.add_header("Authorization", f"Bearer {api_key}")
    req.add_header("Content-Type", "application/json")
    try:
        with request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
            resp_text = resp.read().decode("utf-8", errors="ignore")
            j = json.loads(resp_text)
            return (j.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    except error.HTTPError as e:
        msg = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Groq API HTTPError {e.code}: {msg}") from None
    except error.URLError as e:
        raise RuntimeError(f"Groq API URLError: {e}") from None

# ----------------------------
# Main
# ----------------------------
def main():
    # Check KB file
    if not os.path.isfile(KB_PATH):
        print(f"ERROR: Knowledge base file '{KB_PATH}' not found in current folder: {os.getcwd()}", file=sys.stderr)
        sys.exit(1)

    # Load and chunk KB
    with open(KB_PATH, "r", encoding="utf-8") as f:
        raw_text = normalize_ws(f.read())
    chunks = chunk_text(raw_text, max_tokens=MAX_CHUNK_TOKENS, overlap=CHUNK_OVERLAP)
    if not chunks:
        print("ERROR: KB appears empty after chunking.", file=sys.stderr)
        sys.exit(1)

    # Build index
    index = TFIDFIndex(chunks)

    # Ask user for a question
    print("Enter your question (press Enter to submit):")
    question = input("> ").strip()
    if not question:
        print("No question provided. Exiting.")
        sys.exit(0)

    # Retrieve
    hits = index.retrieve(question, k=TOP_K)

    # Show retrieved
    print("\n==== Retrieved Chunks (Top {}) ====".format(TOP_K))
    for rank, (idx, sc) in enumerate(hits, start=1):
        preview = chunks[idx][:180].replace("\n", " ")
        print(f"{rank}. idx={idx} | score={sc:.4f} | {preview}...")

    # Build context with inline citations
    context_blocks = []
    for idx, sc in hits:
        context_blocks.append(f"[#{idx}] {chunks[idx]}")
    context_text = "\n\n".join(context_blocks) if context_blocks else "No context."

    # Prepare prompts
    system_prompt = (
        "You are a precise legal summarisation assistant. Use ONLY the provided context. "
        "Answer concisely and include inline citations using [#idx] that refer to the chunk indices shown in context. "
        "If the answer is not supported by the context, explicitly say so."
    )
    user_prompt = (
        f"Question:\n{question}\n\n"
        f"Context (each chunk is prefixed with a citation index like [#idx]):\n{context_text}\n\n"
        "Instructions:\n"
        "- Provide a brief, well-structured summary (bullet points are fine).\n"
        "- Cite claims with [#idx] from the context.\n"
        "- Do not invent facts outside the context.\n"
    )

    # Call Groq
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("\nWARNING: GROQ_API_KEY not set. Showing retrieved context only.\n")
        print(context_text)
        sys.exit(0)

    try:
        answer = call_groq_chat(
            api_key=api_key,
            model=MODEL,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=0.2,
            max_tokens=700
        )
    except Exception as e:
        print(f"\nERROR calling Groq API: {e}\n", file=sys.stderr)
        print("Showing retrieved context only:\n")
        print(context_text)
        sys.exit(1)

    print("\n==== Answer ====\n")
    print(answer)

if __name__ == "__main__":
    main()

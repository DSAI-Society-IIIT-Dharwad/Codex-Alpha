import os
import re
import json
import time
import pandas as pd
import requests
from bs4 import BeautifulSoup

# LangChain + Chroma
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LangChain LLM (Grok / xAI)
from langchain_community.chat_models import ChatXAI
from langchain.schema import SystemMessage, HumanMessage

# ----------------------------
# Config
# ----------------------------
CSV_PATH = "kaanoon_titles_links.csv"   # columns: Title, Full Link
CHROMA_DIR = "./chroma_kaanoon_rag"
TITLE_COLLECTION_NAME = "kaanoon_titles"

TITLE_COL = "Title"
URL_COL = "Full Link"

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)

# Model + embeddings
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

# Grok LLM (requires XAI_API_KEY env)
llm = ChatXAI(model="grok-2-latest", temperature=0.2, max_tokens=800)

# ----------------------------
# Utilities
# ----------------------------
def clean_text(s: str) -> str:
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def fetch_html(url: str, timeout: int = 25):
    try:
        r = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=timeout)
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        pass
    return None

def extract_main_text(html: str):
    soup = BeautifulSoup(html, "lxml")
    # Title try
    page_title = None
    if soup.title and soup.title.text:
        page_title = clean_text(soup.title.text)
    h1 = soup.find("h1")
    if h1 and h1.get_text(strip=True):
        page_title = clean_text(h1.get_text(strip=True)) or page_title

    # Main content heuristics
    candidates = []
    for selector in [
        {"name": "article"},
        {"name": "div", "class_": re.compile(r"(content|post|question|answer|article|body|main)", re.I)},
        {"name": "section", "class_": re.compile(r"(content|main|article|details)", re.I)},
    ]:
        nodes = soup.find_all(selector["name"], class_=selector.get("class_"))
        for node in nodes:
            txt = node.get_text(" ", strip=True)
            if txt and len(txt.split()) > 50:
                candidates.append(txt)
    if not candidates:
        paras = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        big = " ".join([p for p in paras if p])
        if len(big.split()) > 30:
            candidates.append(big)

    main_text = max(candidates, key=lambda x: len(x), default="").strip()
    return page_title, main_text

# ----------------------------
# Build Chroma for titles
# ----------------------------
def build_title_index(csv_path: str):
    os.makedirs(CHROMA_DIR, exist_ok=True)
    df = pd.read_csv(csv_path)
    if TITLE_COL not in df.columns or URL_COL not in df.columns:
        raise ValueError(f"CSV must contain '{TITLE_COL}' and '{URL_COL}' columns")

    df = df[[TITLE_COL, URL_COL]].dropna().drop_duplicates()
    df[TITLE_COL] = df[TITLE_COL].apply(clean_text)
    df[URL_COL] = df[URL_COL].apply(clean_text)

    # Create / load collection
    vs = Chroma(
        collection_name=TITLE_COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR,
    )

    # Avoid duplicate growth across runs by checking count and adding only if empty
    if vs._collection.count() == 0:
        docs = df[TITLE_COL].tolist()
        metadatas = [{"url": u, "title": t} for t, u in zip(df[TITLE_COL], df[URL_COL])]
        ids = [f"t_{i}" for i in range(len(docs))]
        vs.add_texts(texts=docs, metadatas=metadatas, ids=ids)
        vs.persist()
    return vs

# ----------------------------
# Step 1: retrieve top-10 titles
# ----------------------------
def retrieve_top_titles(vs: Chroma, query: str, k: int = 10):
    docs = vs.similarity_search_with_score(query, k=k)
    # docs: list of (Document, score) where lower score is better (distance)
    out = []
    for doc, dist in docs:
        url = doc.metadata.get("url", "")
        title = doc.metadata.get("title", doc.page_content)
        sim = 1 - float(dist)  # approximate similarity
        out.append({"title": title, "url": url, "score": sim})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out

# ----------------------------
# Step 2: LLM rerank top-10
# ----------------------------
def llm_pick_best(query: str, candidates: list):
    # Simple instruction to choose the best matching title by semantics
    choices = "\n".join([f"- {c['title']} | {c['url']}" for c in candidates])
    messages = [
        SystemMessage(content="You select the single most relevant link for the user's query based on semantic alignment, specificity, and clarity."),
        HumanMessage(content=f"Query: {query}\n\nTop candidates:\n{choices}\n\nReturn ONLY the best URL.")
    ]
    resp = llm(messages)
    url = resp.content.strip()
    # If LLM returns title lines, try to extract the last URL-like token
    m = re.search(r"https?://\S+", url)
    return m.group(0) if m else url

# ----------------------------
# Step 3: scrape, chunk, vectorize content
# ----------------------------
def build_ephemeral_content_index(page_text: str, persist_dir: str = None):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = splitter.split_text(page_text)
    if not chunks:
        return None, []

    vs = Chroma(
        collection_name="page_chunks",
        embedding_function=embeddings,
        persist_directory=persist_dir,  # can be None for in-memory
    )
    ids = [f"c_{i}" for i in range(len(chunks))]
    vs.add_texts(texts=chunks, metadatas=[{} for _ in chunks], ids=ids)
    return vs, chunks

# ----------------------------
# Step 4: RAG answer with Grok
# ----------------------------
def rag_answer(query: str, url: str, page_title: str, chunk_vs: Chroma, top_k: int = 5):
    rel = chunk_vs.similarity_search(query, k=top_k)
    context = "\n\n".join([f"[Chunk {i+1}]\n{d.page_content}" for i, d in enumerate(rel)])
    prompt = f"""
You are a legal Q&A assistant. Use the provided context to answer the user's query precisely.
Cite short quotes and include the source link at the end.

Context:
{context}

User query: {query}

Instructions:
- Answer concisely with concrete facts.
- Quote the most relevant lines in double quotes.
- Add 'Source: {url}' at the end.
"""
    messages = [
        SystemMessage(content="Be precise, grounded, and cite exact quotes from the context."),
        HumanMessage(content=prompt)
    ]
    resp = llm(messages)
    return resp.content.strip()

# ----------------------------
# Orchestrator
# ----------------------------
def search_titles_scrape_and_rag(query: str):
    # 1) Build/load title store
    title_vs = build_title_index(CSV_PATH)

    # 2) Retrieve + rerank
    top10 = retrieve_top_titles(title_vs, query, k=10)
    if not top10:
        return {"error": "No matches in title store", "query": query}
    best_url = llm_pick_best(query, top10)

    # 3) Fetch + parse
    html = fetch_html(best_url)
    if not html:
        return {"error": "Failed to fetch selected URL", "selected_url": best_url}

    page_title, main_text = extract_main_text(html)
    if not main_text:
        return {"error": "No extractable text on page", "selected_url": best_url}

    # 4) Ephemeral chunk store and RAG
    chunk_vs, chunks = build_ephemeral_content_index(main_text, persist_dir=None)
    if not chunk_vs:
        return {"error": "Failed to build chunk index", "selected_url": best_url}

    answer = rag_answer(query, best_url, page_title or "", chunk_vs, top_k=5)

    return {
        "query": query,
        "selection": {"url": best_url, "page_title": page_title},
        "top10": top10,
        "answer": answer
    }

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    q = input("Enter your legal question/query: ").strip()
    result = search_titles_scrape_and_rag(q)
    print(json.dumps(result, ensure_ascii=False, indent=2))

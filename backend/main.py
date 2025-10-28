import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.responses import JSONResponse
import uvicorn

HERE = Path(__file__).parent.resolve()

SEARCH_SCRIPT = HERE / "search.py"
SCRAPER_SCRIPT = HERE / "scraper_kb.py"
SUMMARIZER_SCRIPT = HERE / "summarization_agent.py"

SEARCH_RESULTS_JSON = HERE / "search_results.json"
KB_TXT = HERE / "kaanoon_all_answers.txt"

# ------------------ Utility Functions ------------------ #

def py_exec() -> str:
    """Return the current Python executable for subprocess calls."""
    return sys.executable or "python"

def check_files_exist():
    missing = []
    for p in [SEARCH_SCRIPT, SCRAPER_SCRIPT, SUMMARIZER_SCRIPT]:
        if not p.exists():
            missing.append(p.name)
    if missing:
        raise FileNotFoundError(
            f"Missing required script(s): {', '.join(missing)}. "
            f"Place them next to main.py."
        )

def run_subprocess(cmd, input_text: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a subprocess with optional stdin."""
    result = subprocess.run(
        cmd,
        input=(input_text.encode("utf-8") if input_text else None),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(HERE),
        shell=False,
        check=False,
    )
    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="ignore")
        stdout = result.stdout.decode("utf-8", errors="ignore")
        raise subprocess.CalledProcessError(result.returncode, cmd, output=stdout, stderr=stderr)
    return result

# ------------------ Pipeline Steps ------------------ #

def step_search(question: str):
    print("\n[1/3] ▶ Running search.py …")
    cmd = [py_exec(), str(SEARCH_SCRIPT)]
    cp = run_subprocess(cmd, input_text=question + "\n")
    if not SEARCH_RESULTS_JSON.exists():
        raise FileNotFoundError(f"{SEARCH_RESULTS_JSON.name} not found.")
    return cp.stdout.decode("utf-8", errors="ignore")

def step_scrape_build_kb():
    print("\n[2/3] ▶ Running scraper_kb.py …")
    cmd = [py_exec(), str(SCRAPER_SCRIPT)]
    cp = run_subprocess(cmd)
    if not KB_TXT.exists():
        raise FileNotFoundError(f"{KB_TXT.name} not found.")
    return cp.stdout.decode("utf-8", errors="ignore")

def step_summarize(question: str):
    print("\n[3/3] ▶ Running summarization_agent.py …")
    cmd = [py_exec(), str(SUMMARIZER_SCRIPT)]
    cp = run_subprocess(cmd, input_text=question + "\n")
    return cp.stdout.decode("utf-8", errors="ignore")

# ------------------ FastAPI Integration ------------------ #

app = FastAPI(title="Legal AI Chatbot Pipeline")

class Query(BaseModel):
    question: str

@app.post("/ask")
async def ask_question(data: Query):
    """Run the 3-step pipeline and return summarized response."""
    question = data.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    try:
        check_files_exist()
        step_search(question)
        step_scrape_build_kb()
        summary = step_summarize(question)
        return JSONResponse(content={
            "status": "success",
            "question": question,
            "summary": summary.strip()
        })
    except Exception as e:
        return JSONResponse(content={
            "status": "error",
            "error": str(e)
        }, status_code=500)

# ------------------ CLI Mode ------------------ #

def cli_main():
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator + FastAPI")
    parser.add_argument("--question", type=str, default="")
    parser.add_argument("--serve", action="store_true", help="Run as FastAPI server.")
    args = parser.parse_args()

    if args.serve:
        uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
        return

    check_files_exist()
    question = args.question.strip()
    if not question:
        question = input("Enter your question: ").strip()
        if not question:
            print("No question provided. Exiting.")
            sys.exit(0)
    try:
        step_search(question)
        step_scrape_build_kb()
        summary = step_summarize(question)
        print("\n✅ Final Summary:\n")
        print(summary)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}\n", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    cli_main()

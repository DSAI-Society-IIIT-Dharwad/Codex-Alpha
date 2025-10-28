import argparse
import os
import sys
import subprocess
from pathlib import Path
from shutil import which
from typing import Optional
from textwrap import dedent

HERE = Path(__file__).parent.resolve()

SEARCH_SCRIPT = HERE / "search.py"
SCRAPER_SCRIPT = HERE / "scraper_kb.py"
SUMMARIZER_SCRIPT = HERE / "summarization_agent.py"

SEARCH_RESULTS_JSON = HERE / "search_results.json"
KB_TXT = HERE / "kaanoon_all_answers.txt"

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

def run_subprocess(
    cmd,
    input_text: Optional[str] = None,
    cwd: Optional[Path] = None
) -> subprocess.CompletedProcess:
    """
    Run a subprocess, optionally providing stdin. Returns CompletedProcess.
    Raises CalledProcessError on non-zero exit codes.
    """
    result = subprocess.run(
        cmd,
        input=(input_text.encode("utf-8") if input_text is not None else None),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=str(cwd or HERE),
        shell=False,
        check=False,
    )
    if result.returncode != 0:
        # Try to give a helpful error message
        stderr = (result.stderr or b"").decode("utf-8", errors="ignore")
        stdout = (result.stdout or b"").decode("utf-8", errors="ignore")
        raise subprocess.CalledProcessError(
            result.returncode,
            cmd,
            output=stdout,
            stderr=stderr
        )
    return result

def step_search(question: str, top_k: int = 10):
    """
    Runs search.py and feeds question to stdin.
    Expects search.py to save search_results.json.
    """
    print("\n[1/3] ▶ Running search.py …")
    if not SEARCH_SCRIPT.exists():
        raise FileNotFoundError(f"{SEARCH_SCRIPT} not found.")

    # search.py is interactive; we pass the question to stdin.
    # It will also print results and write search_results.json.
    cmd = [py_exec(), str(SEARCH_SCRIPT)]
    try:
        cp = run_subprocess(cmd, input_text=question + "\n")
    except subprocess.CalledProcessError as e:
        print("\n--- search.py stdout ---\n" + (e.output or ""), file=sys.stderr)
        print("\n--- search.py stderr ---\n" + (e.stderr or ""), file=sys.stderr)
        raise

    # Optional: show trimmed stdout
    out = cp.stdout.decode("utf-8", errors="ignore")
    print(out)

    if not SEARCH_RESULTS_JSON.exists():
        raise FileNotFoundError(
            f"Expected {SEARCH_RESULTS_JSON.name} not found. "
            f"Ensure search.py writes this file."
        )
    print(f"[OK] search results saved → {SEARCH_RESULTS_JSON.name}")

def step_scrape_build_kb():
    """
    Runs scraper_kb.py to build kaanoon_all_answers.txt from search_results.json.
    """
    print("\n[2/3] ▶ Running scraper_kb.py …")
    if not SCRAPER_SCRIPT.exists():
        raise FileNotFoundError(f"{SCRAPER_SCRIPT} not found.")
    if not SEARCH_RESULTS_JSON.exists():
        raise FileNotFoundError(f"{SEARCH_RESULTS_JSON} not found. Run search step first.")

    cmd = [py_exec(), str(SCRAPER_SCRIPT)]
    try:
        cp = run_subprocess(cmd)
    except subprocess.CalledProcessError as e:
        print("\n--- scraper_kb.py stdout ---\n" + (e.output or ""), file=sys.stderr)
        print("\n--- scraper_kb.py stderr ---\n" + (e.stderr or ""), file=sys.stderr)
        raise

    out = cp.stdout.decode("utf-8", errors="ignore")
    print(out)

    if not KB_TXT.exists():
        raise FileNotFoundError(
            f"Expected {KB_TXT.name} not found. "
            f"Ensure scraper_kb.py writes this file."
        )
    print(f"[OK] knowledge base built → {KB_TXT.name}")

def step_summarize(question: str):
    """
    Runs summarization_agent.py and feeds question to stdin.
    Prints its final output.
    """
    print("\n[3/3] ▶ Running summarization_agent.py …")
    if not SUMMARIZER_SCRIPT.exists():
        raise FileNotFoundError(f"{SUMMARIZER_SCRIPT} not found.")
    if not KB_TXT.exists():
        raise FileNotFoundError(f"{KB_TXT} not found. Run scraper step first.")

    cmd = [py_exec(), str(SUMMARIZER_SCRIPT)]
    try:
        cp = run_subprocess(cmd, input_text=question + "\n")
    except subprocess.CalledProcessError as e:
        print("\n--- summarization_agent.py stdout ---\n" + (e.output or ""), file=sys.stderr)
        print("\n--- summarization_agent.py stderr ---\n" + (e.stderr or ""), file=sys.stderr)
        raise

    out = cp.stdout.decode("utf-8", errors="ignore")
    print(out)

def main():
    parser = argparse.ArgumentParser(description="Pipeline Orchestrator")
    parser.add_argument(
        "--question",
        type=str,
        default="",
        help="Optional: provide the question non-interactively."
    )
    args = parser.parse_args()

    check_files_exist()

    # Ask question if not provided
    question = args.question.strip()
    if not question:
        print("Enter your legal query (the same question will be used for search and summary):")
        question = input("> ").strip()
        if not question:
            print("No question provided. Exiting.")
            sys.exit(0)

    # Run steps
    try:
        step_search(question)
        step_scrape_build_kb()
        step_summarize(question)
    except Exception as e:
        print(f"\n❌ Pipeline failed: {e}\n", file=sys.stderr)
        sys.exit(1)

    print("\n✅ Pipeline completed successfully.")

if __name__ == "__main__":
    main()

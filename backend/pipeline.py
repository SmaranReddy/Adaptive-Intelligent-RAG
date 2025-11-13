# ==========================================
# backend/pipeline.py
# Cinematic logging + progress meters for the research pipeline
# ==========================================

import os
import sys
import time
import traceback
import threading
from typing import TypedDict, List, Dict, Any, Callable
from contextlib import contextmanager
from langgraph.graph import StateGraph
import nodes
from agents.retriever_agent import RetrieverAgent
from agents.answer_agent import AnswerAgent
from agents.critique_agent import CritiqueAgent

# -----------------------------
# Configuration
# -----------------------------
DEBUG = True
SPINNER_INTERVAL = 0.08
PROGRESS_BAR_WIDTH = 30
MAX_PRINT_CHARS = 10000

# ANSI colors
CSI = "\x1b["
RESET = CSI + "0m"
BOLD = CSI + "1m"
GREEN = CSI + "32m"
YELLOW = CSI + "33m"
CYAN = CSI + "36m"
MAGENTA = CSI + "35m"
RED = CSI + "31m"

# -----------------------------
# Logger
# -----------------------------
def _now():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def log(msg: str, level: str = "INFO"):
    if not DEBUG and level == "DEBUG":
        return
    prefix = f"[{_now()}] [{level}]"
    print(f"{BOLD}{CYAN}{prefix}{RESET} {msg}")

# -----------------------------
# Spinner (Threaded)
# -----------------------------
class Spinner:
    _frames = ["|", "/", "-", "\\"]

    def __init__(self, text="working"):
        self.text = text
        self._running = False
        self._thread = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self):
        i = 0
        while self._running:
            frame = self._frames[i % len(self._frames)]
            sys.stdout.write(f"\r{YELLOW}{frame}{RESET} {self.text}")
            sys.stdout.flush()
            time.sleep(SPINNER_INTERVAL)
            i += 1
        sys.stdout.write("\r" + " " * (len(self.text) + 6) + "\r")
        sys.stdout.flush()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=0.5)

# -----------------------------
# Progress Bar
# -----------------------------
def progress_bar(prefix, current, total, width=PROGRESS_BAR_WIDTH):
    if total <= 0:
        bar = "[no items]"
    else:
        frac = current / float(total)
        filled = int(round(width * frac))
        bar = "[" + "█" * filled + "░" * (width - filled) + "]"
        pct = f"{frac*100:6.2f}%"
        bar = f"{bar} {pct}"

    sys.stdout.write(f"\r{BOLD}{prefix}{RESET} {bar}")
    sys.stdout.flush()

    if current >= total:
        sys.stdout.write("\n")

# -----------------------------
# Graph State
# -----------------------------
class GraphState(TypedDict):
    user_query: str
    num_papers: int
    papers: List[Dict[str, Any]]

# -----------------------------
# Node Loader
# -----------------------------
def get_node(name_variants):
    for n in name_variants:
        if hasattr(nodes, n):
            return getattr(nodes, n)
    raise ImportError(f"No matching node found for {name_variants}")

node_search_web = get_node(["node_search_web", "node_search"])
node_download_and_extract = get_node(["node_download_and_extract", "node_download"])
node_clean = get_node(["node_clean", "node_preprocess", "node_clean_text"])
node_chunk = get_node(["node_chunk"])
node_embed = get_node(["node_embed"])
node_index = get_node(["node_index"])

# -----------------------------
# Node Wrapper with logs
# -----------------------------
def wrap_node(fn, display_name=None):
    name = display_name or fn.__name__

    def wrapped(state):
        spinner = Spinner(text=f"[{name}] running...")
        log(f"Starting node: {name}", "DEBUG")
        start = time.time()
        spinner.start()

        try:
            result = fn(state)
            elapsed = time.time() - start
            spinner.stop()
            log(f"Completed {name} in {elapsed:.2f}s", "INFO")
            return result

        except Exception as e:
            spinner.stop()
            tb = traceback.format_exc()
            log(f"Node {name} failed: {e}\n{tb}", "ERROR")
            state.setdefault("_errors", []).append({"node": name, "error": str(e), "trace": tb})
            return state

    return wrapped

# -----------------------------
# Build Graph
# -----------------------------
_graph = StateGraph(GraphState)
_graph.add_node("search_web", wrap_node(node_search_web))
_graph.add_node("download_and_extract", wrap_node(node_download_and_extract))
_graph.add_node("clean", wrap_node(node_clean))
_graph.add_node("chunk", wrap_node(node_chunk))
_graph.add_node("embed", wrap_node(node_embed))
_graph.add_node("index", wrap_node(node_index))

_graph.set_entry_point("search_web")
_graph.add_edge("search_web", "download_and_extract")
_graph.add_edge("download_and_extract", "clean")
_graph.add_edge("clean", "chunk")
_graph.add_edge("chunk", "embed")
_graph.add_edge("embed", "index")
_graph.set_finish_point("index")
_graph = _graph.compile()

# -----------------------------
# Safe Print
# -----------------------------
def safe_print(text, max_chars=MAX_PRINT_CHARS):
    if not text:
        print("(empty)")
        return
    if len(text) <= max_chars:
        print(text)
    else:
        print(text[:max_chars] + "\n\n... (truncated) ...")

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def run_pipeline(query: str, num_papers: int = 5):

    log(f"🚀 Starting indexing phase for query: '{query}'", "INFO")
    overall_start = time.time()

    spinner = Spinner(text="Orchestrating pipeline")
    spinner.start()
    try:
        state = _graph.invoke({"user_query": query, "num_papers": num_papers})
    finally:
        spinner.stop()

    errors = state.get("_errors", [])
    if errors:
        log(f"Pipeline finished with {len(errors)} errors.", "YELLOW")

    papers = state.get("papers", [])
    log(f"Indexed {len(papers)} papers.", "INFO")

    for i, p in enumerate(papers, start=1):
        chunks = len(p.get("chunks", []))
        print(f" {BOLD}{MAGENTA}[{i}/{len(papers)}]{RESET} {p.get('title', 'untitled')} → {chunks} chunks")

    log("Moving to retrieval and summarization phase...", "INFO")

    retriever = RetrieverAgent()
    answerer = AnswerAgent()
    critiquer = CritiqueAgent()

    spinner = Spinner(text="Retrieving relevant chunks")
    spinner.start()
    try:
        retrieved = retriever.retrieve(query)
    finally:
        spinner.stop()

    log(f"Retrieved {len(retrieved)} chunks for answer synthesis.", "INFO")

    # ----------------------------------------
    # 🟨 RAW ANSWER SECTION (NEW)
    # ----------------------------------------
    log("Generating raw answer (LLM)...", "INFO")
    spinner = Spinner(text="LLM Thinking")
    spinner.start()
    try:
        raw_answer = answerer.generate_answer(query, retrieved)
    finally:
        spinner.stop()

    print("\n" + BOLD + YELLOW + "📝 RAW ANSWER (Before Critique):" + RESET + "\n")
    safe_print(raw_answer)

    # ----------------------------------------
    # Critique
    # ----------------------------------------
    log("Refining answer with critique...", "INFO")
    spinner = Spinner(text="Critique running")
    spinner.start()
    try:
        refined = critiquer.critique(raw_answer)
    finally:
        spinner.stop()

    print("\n" + BOLD + GREEN + "🤖 FINAL ANSWER (After Critique):" + RESET + "\n")
    safe_print(refined)

    total_elapsed = time.time() - overall_start
    log(f"Full pipeline finished in {total_elapsed:.2f}s", "INFO")

    # -----------------------------
    # Interactive QA
    # -----------------------------
    print("\n" + BOLD + CYAN + "🗣️ Ask follow-up questions (" 
          "type 'exit' to quit)" + RESET)

    while True:
        user_q = input("\n💬 Your question: ").strip()
        if user_q.lower() in {"exit", "quit"}:
            print("\n👋 Ending session.")
            break

        spinner = Spinner(text="Retrieving context")
        spinner.start()
        try:
            retrieved_q = retriever.retrieve(user_q)
        finally:
            spinner.stop()

        print(BOLD + "Generating answer..." + RESET)
        for i in range(PROGRESS_BAR_WIDTH + 1):
            progress_bar("Answer generation", i, PROGRESS_BAR_WIDTH)
            time.sleep(0.01)

        raw = answerer.generate_answer(user_q, retrieved_q)
        refined = critiquer.critique(raw)

        print("\n" + BOLD + GREEN + "🤖 Answer:" + RESET)
        print("\n" + BOLD + YELLOW + "📝 RAW ANSWER (Before Critique):" + RESET + "\n")
        safe_print(raw)
        print("\n" + BOLD + GREEN + "🤖 REFINED ANSWER (After Critique):" + RESET + "\n")
        safe_print(refined)

# -----------------------------
# ENTRY POINT
# -----------------------------
if __name__ == "__main__":
    try:
        q = input("Enter your research query: ").strip()
        n = input("Number of papers to index (default 5): ").strip()
        n = int(n) if n.isdigit() else 5
        run_pipeline(q, num_papers=n)
    except Exception as e:
        tb = traceback.format_exc()
        log(f"Fatal pipeline error: {e}\n{tb}", "ERROR")
        raise

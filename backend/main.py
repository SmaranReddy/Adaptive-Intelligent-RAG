# run_workflow.py
import sys
import os

# --- Add project root to Python path ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
if root_dir not in sys.path:
    sys.path.append(root_dir)

# --- Absolute imports ---
from backend.agents.planner_agent import PlannerAgent
from backend.agents.retriever_agent import RetrieverAgent
from backend.agents.parser_agent import ParserAgent
from backend.agents.summarizer_agent import SummarizerAgent
from backend.graph_state import GraphState
from langgraph.graph import StateGraph, END

# --- Helper debug function ---
def debug(msg):
    print(f"[DEBUG] {msg}")

# --- 1. Instantiate Agents ---
planner = PlannerAgent()
retriever = RetrieverAgent()
parser = ParserAgent()
summarizer = SummarizerAgent()

# --- 2. Define Graph Node Functions ---
def run_planner(state: GraphState):
    print("\n--- 🧠 ENTERING PLANNER NODE ---")
    user_query = state.get('user_query', "")
    print(f"Input Query: {user_query}")

    if not user_query:
        print("WARNING: Empty user query provided.")
        return {"keywords": []}

    keywords = planner.plan(user_query)
    print(f"Output Keywords: {keywords}")
    print("--- 🧠 EXITING PLANNER NODE ---\n")
    return {"keywords": keywords}


def run_retriever(state: GraphState):
    print("\n--- 📚 ENTERING RETRIEVER NODE ---")
    keywords = state.get('keywords', [])

    if not keywords:
        print("WARNING: No keywords returned from planner.")
        return {"papers": []}

    papers = retriever.search_papers(keywords)
    debug(f"Found {len(papers)} papers.")

    if papers:
        debug("RETRIEVED PAPERS:")
        for i, paper in enumerate(papers):
            title = paper.get('title', 'Untitled')
            debug(f"  {i+1}: {title}")
        print("-----------------------------------")
    else:
        print("WARNING: No papers found.")

    print("--- 📚 EXITING RETRIEVER NODE ---\n")
    return {"papers": papers}


def run_parser(state: GraphState):
    print("\n--- 🗂️ ENTERING PARSER NODE ---")
    papers = state.get('papers', [])

    if not papers:
        print("WARNING: No papers to parse.")
        return {"papers": []}

    papers_with_text = parser.parse_papers(papers)
    debug(f"Completed parsing {len(papers_with_text)} papers.")
    print("--- 🗂️ EXITING PARSER NODE ---\n")
    return {"papers": papers_with_text}


def run_summarizer(state: GraphState):
    print("\n--- ✍️ ENTERING SUMMARIZER NODE ---")
    papers = state.get('papers', [])

    if not papers:
        print("WARNING: No papers to summarize.")
        return {"summaries": []}

    summaries = []
    for i, paper in enumerate(papers):
        content = paper.get("full_text") or paper.get("abstract") or ""
        title = paper.get("title", "Untitled")
        debug(f"Summarizing paper {i+1}: '{title}' (content length: {len(content)})")
        summary = summarizer.summarize(content)
        summaries.append(summary)
        debug(f"Summary generated for paper {i+1}")

    print("--- ✍️ SUMMARIZATION COMPLETE ---\n")
    return {"summaries": summaries}

# --- 3. Build and Compile the Graph ---
workflow = StateGraph(GraphState)

workflow.add_node("planner", run_planner)
workflow.add_node("retriever", run_retriever)
workflow.add_node("parser", run_parser)
workflow.add_node("summarizer", run_summarizer)

workflow.set_entry_point("planner")
workflow.add_edge("planner", "retriever")
workflow.add_edge("retriever", "parser")
workflow.add_edge("parser", "summarizer")
workflow.add_edge("summarizer", END)

app = workflow.compile()

# --- 4. Run function for Streamlit or other frontends ---
def run_graph(user_query: str) -> dict:
    print("\n=== RUNNING GRAPH WORKFLOW ===")
    inputs = {"user_query": user_query}
    final_state = app.invoke(inputs)
    print("=== GRAPH WORKFLOW COMPLETE ===\n")
    return final_state

# --- 5. Main execution for direct run ---
if __name__ == "__main__":
    query = input("Enter your research topic: ")
    final_state = run_graph(query)

    print("\n--- ✅ FINAL SUMMARIES ---")
    papers = final_state.get('papers', [])
    summaries = final_state.get('summaries', [])

    if summaries:
        for i, summary in enumerate(summaries):
            paper_title = papers[i].get('title', 'Untitled') if i < len(papers) else 'Untitled'
            print(f"\n--- Summary for: '{paper_title}' ---")
            print(summary)
    else:
        print("No summaries could be generated.")

import os, re, json
from graph_state import GraphState
from agents.tavily_agent import TavilyAgent
from agents.downloader_agent import DownloaderAgent
from agents.embedder_agent import EmbedderAgent
from agents.index_agent import IndexAgent
from agents.text_utils import preprocess_text, chunk_text

tavily = TavilyAgent()
downloader = DownloaderAgent()
embedder = EmbedderAgent()
indexer = IndexAgent()

# 1️⃣ Search web for relevant papers
def node_search_web(state: GraphState) -> GraphState:
    query = state["user_query"]
    print(f"🔍 Searching Tavily for: {query}")
    papers = tavily.search(query)
    return {"papers": papers}

# 2️⃣ Summarize (optional — can be kept simple)
def node_summarize_abstracts(state: GraphState) -> GraphState:
    return state

# 3️⃣ Download and extract PDFs
def node_download_and_extract(state: GraphState) -> GraphState:
    papers = state["papers"]
    enriched = []
    for item in papers:
        enriched.append(downloader.download_and_extract(item))
    return {"papers": enriched}

# 4️⃣ Preprocess text
def node_preprocess(state: GraphState) -> GraphState:
    for p in state["papers"]:
        p["clean_text"] = preprocess_text(p.get("full_text", ""))
    return state

# 5️⃣ Tokenize (optional)
def node_tokenize(state: GraphState) -> GraphState:
    return state

# 6️⃣ Chunk text
def node_chunk(state: GraphState) -> GraphState:
    for p in state["papers"]:
        p["chunks"] = chunk_text(p.get("clean_text", ""), chunk_size=1000, overlap=150)
    return state

# 7️⃣ Embed chunks
def node_embed(state: GraphState) -> GraphState:
    for p in state["papers"]:
        p["embeddings"] = embedder.embed_chunks(p["chunks"])
    return state

# 8️⃣ Index into Pinecone
def node_index(state: GraphState) -> GraphState:
    for p in state["papers"]:
        indexer.index_chunks(p["title"], p["chunks"], p["embeddings"])
    return state

# 9️⃣ Final summary node (Fixes final_message KeyError)
def node_finalize(state: GraphState) -> GraphState:
    papers = state.get("papers", [])
    msg = f"✅ Indexed {len(papers)} papers.\n"
    for p in papers:
        msg += f"- {p.get('title', 'Untitled')} → {len(p.get('chunks', []))} chunks\n"
    return {"final_message": msg}

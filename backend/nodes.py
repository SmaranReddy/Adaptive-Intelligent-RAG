# backend/nodes.py

from agents.tavily_agent import TavilyAgent
from agents.downloader_agent import DownloaderAgent
from agents.embedder_agent import EmbedderAgent
from agents.index_agent import IndexAgent
from agents.llm_agent import SummarizerAgent
from agents.text_utils import PreprocessingAgent, TokenizerAgent, ChunkingAgent
from graph_state import GraphState

tavily = TavilyAgent()
summarizer = SummarizerAgent()
downloader = DownloaderAgent()
pre = PreprocessingAgent()
tok = TokenizerAgent()
chunker = ChunkingAgent()
embedder = EmbedderAgent()
indexer = IndexAgent()

# 1) Search using Tavily
def node_search_web(state: GraphState) -> GraphState:
    query = state["user_query"]
    num = state.get("num_papers", 3)

    papers = tavily.search(query, max_results=num)
    return {"papers": papers}

# 2) Summarize abstract/snippet
def node_summarize_abstracts(state: GraphState) -> GraphState:
    for p in state["papers"]:
        try:
            p["summary_structured"] = summarizer.summarize(p["abstract"])
        except Exception as e:
            state.setdefault("errors", []).append(str(e))
    return {"papers": state["papers"]}

# 3) Download + Extract text
def node_download_and_extract(state: GraphState) -> GraphState:
    papers = state["papers"]
    for i, p in enumerate(papers):
        papers[i] = downloader.download_and_extract(p)
        if not papers[i].get("full_text"):
            # Fallback to snippet/abstract so downstream doesn’t end up with 0 chunks
            fallback = p.get("abstract") or p.get("summary") or ""
            papers[i]["full_text"] = fallback
            if fallback:
                print(f"↩️ Using fallback snippet for: {p.get('title', 'untitled')}")
            else:
                print(f"⚠️ No text available for: {p.get('title', 'untitled')}")
    return {"papers": papers}


# 4) Preprocess
def node_preprocess(state: GraphState) -> GraphState:
    cleaned = [pre.preprocess(p.get("full_text") or "") for p in state["papers"]]
    return {"cleaned_texts": cleaned}

# 5) Tokenize
def node_tokenize(state: GraphState) -> GraphState:
    counts = [tok.count_tokens(t) for t in state["cleaned_texts"]]
    return {"token_counts": counts}

# 6) Chunk
def node_chunk(state: GraphState) -> GraphState:
    for p, txt in zip(state["papers"], state["cleaned_texts"]):
        p["chunks"] = chunker.chunk_text(txt)
    return {"papers": state["papers"]}

# 7) Embed
def node_embed(state: GraphState) -> GraphState:
    for p in state["papers"]:
        p["chunk_embeddings"] = [
            embedder.embed_text(chunk) for chunk in p["chunks"]
        ]
    return {"papers": state["papers"]}

# 8) Index (minimal metadata)
def node_index(state: GraphState) -> GraphState:
    for p in state["papers"]:
        p["upsert_responses"] = []
        for j, (chunk, emb) in enumerate(zip(p["chunks"], p["chunk_embeddings"]), start=1):
            meta = {
                "title": p["title"],
                "link": p.get("link", ""),
                "chunk_id": j,
                "chunk_text": chunk[:400] + "..."
            }
            resp = indexer.upsert_paper(meta, emb)
            p["upsert_responses"].append(resp)
    return {"papers": state["papers"]}

# 9) Final
def node_finalize(state: GraphState) -> GraphState:
    msg = f"✅ Indexed {len(state['papers'])} papers.\n"
    for p in state["papers"]:
        msg += f"- {p['title']} → {len(p.get('chunks', []))} chunks\n"
    return {"final_message": msg}

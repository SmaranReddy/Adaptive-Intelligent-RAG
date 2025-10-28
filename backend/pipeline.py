# backend/pipeline.py

import re
import numpy as np
from agents.arxiv_agent import ArxivAgent
from agents.downloader_agent import DownloaderAgent
from agents.embedder_agent import EmbedderAgent
from agents.index_agent import IndexAgent
from agents.llm_agent import SummarizerAgent


# ---------------------------------------------------------------------
# 🧹 Helper: Sanitize filenames for Windows
# ---------------------------------------------------------------------
def sanitize_filename(title: str) -> str:
    """Removes invalid characters and extra spaces from filenames."""
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    title = title.replace("\n", " ").replace("\r", " ").strip()
    return " ".join(title.split())  # normalize multiple spaces


# ---------------------------------------------------------------------
# 🚀 Main RAG Pipeline
# ---------------------------------------------------------------------
def main(query: str):
    """Run the full research pipeline for a given query."""
    # Initialize all agents
    arxiv = ArxivAgent()
    downloader = DownloaderAgent()
    embedder = EmbedderAgent()
    indexer = IndexAgent()
    summarizer = SummarizerAgent()

    print(f"\n🔍 Searching ArXiv for: {query}")
    papers = arxiv.fetch_papers(query, max_results=3)

    for paper in papers:
        try:
            print(f"\n📄 Processing: {paper['title']}")
            print("-" * 80)

            # Step 1: Summarize abstract
            summary = summarizer.summarize(paper["abstract"])
            paper["summary_structured"] = summary
            print("🧠 Summarized abstract.")
            print(f"📝 Summary preview:\n{summary[:400]}...\n")

            # Step 2: Sanitize title for file operations
            safe_title = sanitize_filename(paper["title"])
            paper["safe_title"] = safe_title

            # Step 3: Download and extract text
            paper = downloader.download_and_extract(paper)
            if not paper.get("full_text"):
                print("⚠️ Skipping embedding (no text found).")
                continue

            # Step 4: Save summary for inspection
            summary_path = f"{safe_title}.txt"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(summary)
            print(f"💾 Saved summary as {summary_path}")

            # Step 5: Generate embedding (limit text length)
            text_to_embed = paper["full_text"][:5000]
            embedding = embedder.embed_text(text_to_embed)

            # Convert to NumPy array for debug visibility
            vec = np.array(embedding)
            print(f"📊 Embedding generated! Dim={vec.shape[0]}")
            print(f"🔢 First 10 values: {vec[:10]}\n")

            # Step 6: Upsert into vector DB
            resp = indexer.upsert_paper(paper, embedding)
            print("✅ Stored in Pinecone successfully!")
            print(f"📦 Response snippet: {str(resp)[:200]}...\n")

        except Exception as e:
            print(f"❌ Error with {paper['title']}: {e}")
            continue

    print("\n🚀 Pipeline complete!\n")


# ---------------------------------------------------------------------
# 🧑‍💻 Entry Point (Interactive mode)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    user_query = input("Enter your research query: ").strip()
    if user_query:
        main(user_query)
    else:
        print("⚠️ No query entered. Exiting.")

# backend/pipeline.py

import re
import os
import time
import json
import numpy as np
import traceback

# Agents
from agents.downloader_agent import DownloaderAgent
from agents.embedder_agent import EmbedderAgent
from agents.index_agent import IndexAgent
from agents.llm_agent import SummarizerAgent
from agents.tavily_agent import fetch_paper_pdf


# ---------------------------------------------------------------------
# 🧹 Helper: Sanitize filenames
# ---------------------------------------------------------------------
def sanitize_filename(title: str) -> str:
    """Removes invalid characters and extra spaces from filenames."""
    title = re.sub(r'[\\/*?:"<>|]', "", title)
    title = title.replace("\n", " ").replace("\r", " ").strip()
    return " ".join(title.split())


# ---------------------------------------------------------------------
# 🧰 Text Processing, Tokenization, Chunking, Metadata
# ---------------------------------------------------------------------
class PreprocessingAgent:
    def preprocess(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\n+", "\n", text)
        text = re.split(r"\bReferences\b", text, maxsplit=1)[0]
        text = re.sub(r"Figure\s*\d+|Table\s*\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class TokenizerAgent:
    def __init__(self):
        try:
            import tiktoken
            self.enc = tiktoken.encoding_for_model("text-embedding-3-small")
        except Exception:
            self.enc = None

    def count_tokens(self, text: str) -> int:
        if self.enc:
            return len(self.enc.encode(text))
        return len(text.split())

    def truncate(self, text: str, max_tokens: int = 8000) -> str:
        if self.enc:
            tokens = self.enc.encode(text)
            return self.enc.decode(tokens[:max_tokens])
        return " ".join(text.split()[:max_tokens])


class ChunkingAgent:
    def __init__(self, chunk_size=1500, overlap=200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str):
        chunks, start = [], 0
        while start < len(text):
            end = start + self.chunk_size
            chunks.append(text[start:end])
            start += self.chunk_size - self.overlap
        return chunks


def prepare_paper_for_indexing(paper: dict, max_summary_chars: int = 1000):
    """Shrink metadata before Pinecone upsert."""
    safe = dict(paper or {})
    for k in ("full_text", "raw_pdf", "pdf_bytes", "text_chunks", "pages"):
        safe.pop(k, None)
    for key in ("summary_structured", "summary", "abstract"):
        if key in safe:
            val = str(safe[key])
            if len(val) > max_summary_chars:
                val = val[:max_summary_chars] + "...[truncated]"
            safe[key] = val
    return safe


# ---------------------------------------------------------------------
# 🚀 Tavily-Only Research Pipeline (with link key)
# ---------------------------------------------------------------------
def main(query: str):
    print("=" * 100)
    print(f"🧠 Starting Tavily Research Pipeline for query: '{query}'")
    print("=" * 100)

    # Initialize all agents
    print("\n⚙️ Initializing agents...")
    downloader = DownloaderAgent()
    embedder = EmbedderAgent()
    indexer = IndexAgent()
    summarizer = SummarizerAgent()
    preprocessor = PreprocessingAgent()
    tokenizer = TokenizerAgent()
    chunker = ChunkingAgent()
    print("✅ All agents initialized successfully.\n")

    # Step 1: Tavily search
    print(f"🌐 Searching Tavily for: '{query}' ...")
    tavily_pdf_link = fetch_paper_pdf(query, max_results=5)

    if not tavily_pdf_link:
        print("❌ No suitable PDF found by Tavily.")
        return

    print(f"📘 Proceeding with Tavily PDF: {tavily_pdf_link}")

    # Step 2: Download and extract
    try:
        print("📥 Downloading and extracting full text from PDF...")
        paper = {"title": query, "link": tavily_pdf_link, "abstract": ""}
        paper = downloader.download_and_extract(paper)
        if not paper.get("full_text"):
            print("⚠️ No text extracted from the PDF.")
            return

        full = paper["full_text"]
        print(f"📄 Extracted {len(full)} characters.")

        os.makedirs("downloads", exist_ok=True)
        json_path = os.path.join("downloads", f"{sanitize_filename(query)}.json")
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(paper, jf, ensure_ascii=False, indent=2)
        print(f"💾 Saved paper data as JSON: {json_path}")

        # Step 3: Preprocess
        clean = preprocessor.preprocess(full)
        print(f"🧹 Cleaned text length: {len(clean)} chars.")

        # Step 4: Tokenize and truncate
        tokens = tokenizer.count_tokens(clean)
        print(f"🧮 Token count: {tokens}")
        if tokens > 8000:
            clean = tokenizer.truncate(clean)
            print("⚠️ Text truncated to 8000 tokens for safety.")

        # Step 5: Chunk
        chunks = chunker.chunk_text(clean)
        print(f"✂️ Split into {len(chunks)} chunks for embedding.\n")

        # Step 6: Optional summary (for metadata)
        print("✏️ Summarizing document introduction using LLM...")
        intro_text = clean[:2000]
        summary = summarizer.summarize(intro_text)
        paper["summary_structured"] = summary
        print(f"✅ Summary created. Preview:\n{summary[:400]}...\n")

        # Step 7: Embed + Index
        for j, chunk in enumerate(chunks, start=1):
            emb = embedder.embed_text(chunk)
            paper_meta = prepare_paper_for_indexing(paper, 1000)
            paper_meta["chunk_id"] = j
            paper_meta["chunk_text"] = chunk[:400] + "..." if len(chunk) > 400 else chunk
            resp = indexer.upsert_paper(paper_meta, emb)
            print(f"✅ Upserted chunk {j}. Resp: {str(resp)[:150]}")

        print("\n🚀 Tavily Research Pipeline completed successfully!\n")

    except Exception as e:
        print(f"❌ Error in pipeline: {e}")
        traceback.print_exc()


# ---------------------------------------------------------------------
# 🧑‍💻 Entry Point
# ---------------------------------------------------------------------
if __name__ == "__main__":
    query = input("Enter your research query: ").strip()
    if not query:
        print("⚠️ No query entered. Exiting.")
    else:
        main(query)

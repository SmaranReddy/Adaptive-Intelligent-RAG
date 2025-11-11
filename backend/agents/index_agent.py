# backend/agents/index_agent.py

from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

class IndexAgent:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing PINECONE_API_KEY in environment.")
        
        print("🔗 Connecting to Pinecone index...")
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(
            host="re-search-02vwk3u.svc.aped-4627-b74a.pinecone.io"
        )
        print("✅ Pinecone index connection established.\n")

    def upsert_paper(self, paper, embedding):
        """
        Upload minimal metadata to avoid Pinecone 40KB limit.
        """

        vector = {
            "id": f"{paper['title'].replace(' ', '_')[:80]}_{paper['chunk_id']}",
            "values": embedding,
            "metadata": {
                "title": paper["title"],
                "link": paper.get("link", ""),
                "chunk_id": paper.get("chunk_id", 0),
                "chunk_text": paper.get("chunk_text", "")[:500]  # safe preview
            }
        }

        print(f"📤 Uploading vector for: {paper['title'][:80]} (chunk {paper['chunk_id']})")

        response = self.index.upsert(
            namespace="research-papers",
            vectors=[vector]
        )

        print(f"✅ Upsert OK for chunk {paper['chunk_id']}\n")
        return response

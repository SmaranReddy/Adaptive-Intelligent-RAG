# backend/agents/index_agent.py
from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

class IndexAgent:
    def __init__(self):
        api_key = os.getenv("PINECONE_API_KEY")
        if not api_key:
            raise ValueError("Missing PINECONE_API_KEY")
        self.pc = Pinecone(api_key=api_key)
        self.index = self.pc.Index(
            host="re-search-02vwk3u.svc.aped-4627-b74a.pinecone.io"
        )

    def upsert_paper(self, paper, embedding):
        vector = {
            "id": paper["title"].replace(" ", "_")[:80],
            "values": embedding,
            "metadata": {
                "title": paper["title"],
                "authors": paper["authors"],
                "abstract": paper["abstract"],
                "link": paper["link"],
                "published": paper["published"]
            }
        }
        response = self.index.upsert(
            namespace="research-papers",
            vectors=[vector]
        )
        return response

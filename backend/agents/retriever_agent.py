# ==========================================
# backend/agents/retriever_agent.py (Fixed)
# ==========================================
import google.generativeai as genai
from pinecone import Pinecone
import os

class RetrieverAgent:
    def __init__(self):
        # Initialize Pinecone
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index = pc.Index("research-papers")
        print("✅ RetrieverAgent connected to Pinecone index.")

        # Initialize Google embedding model
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.embed_model = "models/text-embedding-004"
        print("✅ Google embedding model initialized for retrieval.")

    def embed_query(self, text: str):
        """Convert query text into vector embedding"""
        result = genai.embed_content(model=self.embed_model, content=text)
        return result["embedding"]

    def retrieve(self, query: str, top_k: int = 5):
        """Retrieve top relevant chunks for the query"""
        query_vector = self.embed_query(query)  # ✅ FIXED — embed before querying

        response = self.index.query(
            namespace="research-papers",  # matches your index namespace
            vector=query_vector,
            top_k=top_k,
            include_metadata=True
        )

        results = []
        for match in response.get("matches", []):
            metadata = match.get("metadata", {})
            text = metadata.get("text", "")
            if text.strip():
                results.append(text)
        return results

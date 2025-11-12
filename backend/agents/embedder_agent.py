# ==========================================
# backend/agents/embedder_agent.py
# ==========================================
import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
import numpy as np

class EmbedderAgent:
    """
    Uses Google's Gemini embedding model (text-embedding-004)
    to embed research paper chunks for Pinecone indexing.
    Supports both single and batch embedding for high speed.
    """

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("❌ GOOGLE_API_KEY missing in environment variables.")
        self.client = genai.Client(api_key=api_key)
        print("✅ Google embedding model (text-embedding-004) initialized")

    # ======================================================
    # 🧠 Embed a single text string
    # ======================================================
    def embed_text(self, text: str) -> list[float]:
        """
        Generates an embedding for a single text chunk.
        """
        if not text.strip():
            return []

        try:
            response = self.client.models.embed_content(
                model="text-embedding-004",
                contents=text,
                config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
            )
            vector = np.array(response.embeddings[0].values).tolist()
            return vector
        except Exception as e:
            print(f"⚠️ Embedding failed: {e}")
            return []

    # ======================================================
    # ⚡ Batch embed all chunks at once (faster)
    # ======================================================
    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        """
        Embeds a list of text chunks using a single batch API call.
        Much faster and cheaper than calling embed_text() for each chunk.
        """
        if not chunks:
            return []

        # Limit per API call (Google supports up to ~1000 items)
        BATCH_SIZE = 50
        all_embeddings = []

        print(f"🚀 Starting batch embedding for {len(chunks)} chunks...")

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            print(f"🧠 Embedding batch {i//BATCH_SIZE + 1} ({len(batch)} chunks)...")

            try:
                response = self.client.models.embed_content(
                    model="text-embedding-004",
                    contents=batch,
                    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
                )
                batch_embeddings = [np.array(e.values).tolist() for e in response.embeddings]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                print(f"⚠️ Batch embedding failed at batch {i//BATCH_SIZE + 1}: {e}")
                all_embeddings.extend([[] for _ in batch])

        print(f"✅ Completed embedding of {len(all_embeddings)} chunks.")
        return all_embeddings

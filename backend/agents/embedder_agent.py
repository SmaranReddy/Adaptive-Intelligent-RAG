# backend/agents/embedder_agent.py
from google import genai
from google.genai import types
from dotenv import load_dotenv
import numpy as np
import os

load_dotenv()

class EmbedderAgent:
    def __init__(self):
        self.client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

    def embed_text(self, text):
        response = self.client.models.embed_content(
            model="text-embedding-004",
            contents=text,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
        )
        return np.array(response.embeddings[0].values).tolist()

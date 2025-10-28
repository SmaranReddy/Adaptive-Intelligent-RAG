from google import genai
from google.genai import types
import numpy as np
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# Input text
text = "What is the meaning of life?"

# Use the 1024-dim embedding model
response = client.models.embed_content(
    model="text-embedding-004",
    contents=text,
    config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY")
)

# Extract and print
embedding = np.array(response.embeddings[0].values)
print(f"✅ Embedding dimension: {embedding.shape[0]}")
print(embedding[:10])  # print first 10 values only

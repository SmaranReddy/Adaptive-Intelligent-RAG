import os
from dotenv import load_dotenv
from pinecone import Pinecone

# -----------------------------
# Step 0: Load environment variables
# -----------------------------
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")
if not api_key:
    raise ValueError("❌ PINECONE_API_KEY not found in .env")

# -----------------------------
# Step 1: Initialize Pinecone client
# -----------------------------
pc = Pinecone(api_key=api_key)
print("✅ Pinecone client initialized successfully")

# -----------------------------
# Step 2: Define index name
# -----------------------------
index_name = "quickstart-py"

# -----------------------------
# Step 3: Check if index exists; create if not
# -----------------------------
if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",
        region="us-east-1",
        embed={
            "model": "text-embedding-004",  # Google embedding model
            "field_map": {"text": "chunk_text"}
        }
    )
    print(f"✅ Created Pinecone index '{index_name}' with embedding model text-embedding-004")
else:
    print(f"ℹ️ Index '{index_name}' already exists")

print("✅ Pinecone index setup complete.")

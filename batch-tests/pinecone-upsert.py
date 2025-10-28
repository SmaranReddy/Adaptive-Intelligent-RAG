from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os
import sys

# -----------------------------
# Step 0: Load environment variables
# -----------------------------
load_dotenv()

# Get Pinecone API key from .env
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    print("❌ Missing PINECONE_API_KEY in .env file")
    sys.exit(1)

# -----------------------------
# Step 1: Initialize Pinecone client
# -----------------------------
try:
    pc = Pinecone(api_key=api_key)
    print("✅ Pinecone client initialized successfully")
except Exception as e:
    print("❌ Failed to initialize Pinecone client:", e)
    sys.exit(1)

# -----------------------------
# Step 2: Connect to your index
# -----------------------------
index_host = "quickstart-py-02vwk3u.svc.aped-4627-b74a.pinecone.io"  # Replace with actual host

try:
    index = pc.Index(host=index_host)
    print(f"✅ Connected to index at host: {index_host}")
except Exception as e:
    print("❌ Failed to connect to index:", e)
    sys.exit(1)

# -----------------------------
# Step 3: Debug - List available indexes
# -----------------------------
try:
    indexes = pc.list_indexes()
    print("ℹ️ Available indexes:", indexes)
except Exception as e:
    print("❌ Failed to list indexes:", e)

# -----------------------------
# Step 4: Delete existing chunks
# -----------------------------
try:
    deleted = index.delete(
        namespace="example-namespace",
        filter={"document_id": {"$eq": "document1"}}
    )
    print("✅ Deleted existing chunks for document1:", deleted)
except Exception as e:
    print("❌ Failed to delete existing chunks:", e)

# -----------------------------
# Step 5: Upsert updated chunks
# -----------------------------
vectors_to_upsert = [
    {
        "id": "document1#first",
        "values": [0.01, 0.02, 0.03] + [0.0] * 1021,
        "metadata": {
            "document_id": "document44",
            "document_title": "Introduction to Vector Databases - Updated Edition",
            "chunk_number": 1,
            "chunk_text": "First chunk with new content...",
            "document_url": "https://example.com/docs/document1",
            "created_at": "2024-02-15",
            "document_type": "tutorial",
            "version": "2.1"
        }
    },
    {
        "id": "document1#second",
        "values": [0.11, -0.05, 0.07] + [0.0] * 1021,
        "metadata": {
            "document_id": "documejhbjbhbhjnt1",
            "document_title": "Introduction to Vector Databases - Updated Edition",
            "chunk_number": 2,
            "chunk_text": "Second chunk with new content...",
            "document_url": "https://example.com/docs/document1",
            "created_at": "2024-02-15",
            "document_type": "tutorial",
            "version": "2.1"
        }
    }
]

try:
    response = index.upsert(
        namespace="example-namespace",
        vectors=vectors_to_upsert
    )
    print("✅ Successfully upserted document1 chunks:", response)
except Exception as e:
    print("❌ Failed to upsert vectors:", e)

# -----------------------------
# Step 6: Debug - Count vectors in namespace
# -----------------------------
try:
    stats = index.describe_index_stats(
        namespace="example-namespace"
    )
    print("ℹ️ Index stats for namespace 'example-namespace':", stats)
except Exception as e:
    print("❌ Failed to get index stats:", e)
# -----------------------------
# End of Script
# -----------------------------
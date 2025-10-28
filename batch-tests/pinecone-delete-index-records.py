from pinecone.grpc import PineconeGRPC as Pinecone
from dotenv import load_dotenv
import os
import sys

# Load environment variables
load_dotenv()
api_key = os.getenv("PINECONE_API_KEY")

if not api_key:
    print("❌ Missing PINECONE_API_KEY in .env file")
    sys.exit(1)

# Initialize Pinecone client
try:
    pc = Pinecone(api_key=api_key)
    print("✅ Pinecone client initialized successfully")
except Exception as e:
    print("❌ Failed to initialize Pinecone client:", e)
    sys.exit(1)

# Connect to index
index_host = "re-search-02vwk3u.svc.aped-4627-b74a.pinecone.io"
try:
    index = pc.Index(host=index_host)
    print(f"✅ Connected to index at host: {index_host}")
except Exception as e:
    print("❌ Failed to connect to index:", e)
    sys.exit(1)

# Get existing namespaces
try:
    stats = index.describe_index_stats()
    namespaces = list(stats["namespaces"].keys())
    print("ℹ️ Existing namespaces:", namespaces)
except Exception as e:
    print("❌ Failed to get index stats:", e)
    sys.exit(1)

# Delete all vectors from each namespace
for ns in namespaces:
    try:
        response = index.delete(
            namespace=ns,
            filter={"document_id": {"$exists": True}}  # Matches all vectors with a document_id
        )
        print(f"✅ Deleted all vectors from namespace '{ns}':", response)
    except Exception as e:
        print(f"❌ Failed to delete vectors from namespace '{ns}':", e)

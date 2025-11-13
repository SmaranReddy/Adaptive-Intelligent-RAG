import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from pinecone import Pinecone
from groq import Groq

# ------------------------------------
# Pretty print helper
# ------------------------------------
def pretty(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False))


# ------------------------------------
# Load env
# ------------------------------------
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not PINECONE_API_KEY or not GOOGLE_API_KEY or not GROQ_API_KEY:
    print("❌ Missing one or more API keys in .env")
    exit(1)


# ------------------------------------
# MAIN SCRIPT
# ------------------------------------
if __name__ == "__main__":
    print("\n=== 🔧 FULL RAG TEST (Google Embedding + Pinecone Retrieval + Groq Refinement) ===")

    INDEX_NAME = input("Enter Pinecone Index Name: ").strip()
    NAMESPACE = input("Enter Namespace (default = default): ").strip() or "default"
    QUERY = input("\nEnter your query: ").strip()

    # ------------------------------------
    # Setup Pinecone
    # ------------------------------------
    print("\n🔗 Connecting to Pinecone...")
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    print(f"✅ Connected to index: {INDEX_NAME}")

    # ------------------------------------
    # Setup Google Embedding
    # ------------------------------------
    genai.configure(api_key=GOOGLE_API_KEY)
    EMBEDDING_MODEL = "models/text-embedding-004"
    print("✅ Google embedding model ready")

    # ------------------------------------
    # Embed Query
    # ------------------------------------
    print("\n📘 Generating query embedding...")
    embed_res = genai.embed_content(
        model=EMBEDDING_MODEL,
        content=QUERY
    )
    vector = embed_res["embedding"]
    print("✅ Embedding generated")

    print("\n🔹 QUERY:", QUERY)
    print("\n🔹 EMBEDDING (first 10 dims):", vector[:10])

    # ------------------------------------
    # Retrieval
    # ------------------------------------
    print("\n🔍 Running similarity search...")
    response = index.query(
        namespace=NAMESPACE,
        vector=vector,
        top_k=5,
        include_metadata=True
    )

    matches = response.get("matches", [])
    print(f"✅ Retrieved {len(matches)} matches")

    print("\n========= 🔎 RETRIEVED CHUNKS =========")
    context_blocks = []
    for i, m in enumerate(matches, start=1):
        meta = m.get("metadata", {}) or {}
        text = meta.get("text", "")
        title = meta.get("title", "no-title")

        clean_text = text[:250].replace("\n", " ")
        print(f"\n--- Chunk #{i} ---")
        print(f"ID: {m.get('id')}")
        print(f"Score: {m.get('score'):.4f}")
        print(f"Title: {title}")
        print(f"Snippet: {clean_text}...")

        # Full text for refinement
        context_blocks.append(f"### {title}\n{text}")

    context = "\n\n".join(context_blocks)

    # ------------------------------------
    # GROQ REFINER
    # ------------------------------------
    print("\n✨ Sending to Groq Llama3-70B for refinement...")

    groq = Groq(api_key=GROQ_API_KEY)

    prompt = f"""
You are a research assistant. Your job is to answer the user's question using ONLY the context provided.

User Query:
{QUERY}

Context:
{context}

Guidelines:
- Use only facts available in the retrieved chunks.
- No hallucination.
- If information is missing, say so.
- Provide a clear, structured, research-grade answer.
"""

    response = groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=800
    )

    refined_answer = response.choices[0].message.content

    # ------------------------------------
    # FINAL ANSWER
    # ------------------------------------
    print("\n================= 🧠 FINAL ANSWER (GROQ) =================")
    print(refined_answer)
    print("==========================================================\n")

# ==========================================
# backend/agents/answer_agent.py (RAG-enabled)
# ==========================================
import os
from groq import Groq
from dotenv import load_dotenv

class AnswerAgent:
    """
    Synthesizes a final answer using Groq LLM with full RAG context.
    """
    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing GROQ_API_KEY in .env file")

        self.client = Groq(api_key=api_key)
        self.model = "llama-3.1-8b-instant"   # 🔥 best Groq model
        print("✅ AnswerAgent ready (Groq RAG engine).")

    def generate_answer(self, query: str, retrieved_matches: list) -> str:
        """
        retrieved_matches: output from RetrieverAgent.retrieve()
        """

        if not retrieved_matches:
            return "⚠️ No relevant information retrieved."

        # Build context in the exact format as standalone RAG test
        context_blocks = []
        for m in retrieved_matches:
            title = m["metadata"].get("title", "Unknown Title")
            text = m["text"]
            context_blocks.append(f"### {title}\n{text}")

        context = "\n\n".join(context_blocks)

        prompt = f"""
You are a research assistant. Your job is to answer the user's question using ONLY the context provided.

User Query:
{query}

Context:
{context}

Guidelines:
- Use only facts available in the retrieved chunks.
- No hallucination.
- If information is missing, say so.
- Provide a clear, structured, research-grade answer.
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=900,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            print(f"⚠️ Groq request failed: {e}")
            return "⚠️ Could not generate answer due to LLM error."

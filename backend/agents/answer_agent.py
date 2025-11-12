# ==========================================
# backend/agents/answer_agent.py
# ==========================================
import os
from groq import Groq
from dotenv import load_dotenv

class AnswerAgent:
    """
    Uses Groq's LLM to synthesize a coherent answer
    from the retrieved text context.
    """

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing GROQ_API_KEY in .env file")

        self.client = Groq(api_key=api_key)
        print("✅ AnswerAgent ready (Groq LLM).")

    def generate_answer(self, query: str, context: list[str]) -> str:
        """
        Combine retrieved chunks into a final answer using Groq.
        """
        context_text = "\n\n".join(context)
        if len(context_text) > 15000:  # keep prompt size manageable
            context_text = context_text[:15000]

        prompt = f"""
        You are an expert AI research summarizer.
        The following are extracted research paper snippets about: "{query}".

        Context:
        {context_text}

        Based on the above, write a concise, factual, and well-structured explanation.
        - Do NOT hallucinate or fabricate details.
        - Include paper insights if found.
        - Write clearly in academic tone.
        """

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",   # ✅ UPDATED MODEL
                messages=[
                    {"role": "system", "content": "You are a helpful academic AI assistant."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=1000,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Groq request failed: {e}")
            return "⚠️ Could not generate answer due to LLM error."

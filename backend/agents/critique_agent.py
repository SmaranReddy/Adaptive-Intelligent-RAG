# ==========================================
# backend/agents/critique_agent.py
# ==========================================
import os
from groq import Groq
from dotenv import load_dotenv


class CritiqueAgent:
    """
    Uses Groq LLM to critique and improve the AnswerAgent's output.
    Provides structured feedback on factual accuracy, clarity, and completeness.
    """

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing GROQ_API_KEY in .env file")

        self.client = Groq(api_key=api_key)
        print("✅ CritiqueAgent ready.")

    def critique(self, answer: str) -> str:
        """
        Takes the raw answer from the AnswerAgent and produces
        an improved, verified, and refined response.
        """

        prompt = f"""
        You are an expert academic reviewer.

        Below is an AI-generated research summary that may contain inaccuracies or weak reasoning.
        Your job is to:
        1. Identify any factual inconsistencies or hallucinations.
        2. Improve clarity and flow.
        3. Make it sound like a concise academic abstract.
        4. If the text is already good, briefly confirm its quality.

        --- BEGIN ANSWER ---
        {answer}
        --- END ANSWER ---

        Now provide your improved version below:
        """

        try:
            response = self.client.chat.completions.create(
                model="llama-3.1-8b-instant",  # ✅ Updated model
                messages=[
                    {"role": "system", "content": "You are an expert academic reviewer."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=1000,
            )

            improved_text = response.choices[0].message.content.strip()
            print("\n✅ Critique Completed Successfully.\n")
            return improved_text

        except Exception as e:
            print(f"⚠️ CritiqueAgent failed: {e}")
            return "⚠️ Could not perform critique due to LLM error."

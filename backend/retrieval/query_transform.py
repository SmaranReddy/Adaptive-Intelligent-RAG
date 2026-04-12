import os
from concurrent.futures import ThreadPoolExecutor
from groq import Groq
from dotenv import load_dotenv

load_dotenv()


class QueryTransformer:
    """
    Generates multiple query variations for multi-query retrieval.
    Returns a list: [original_query, HyDE_rewrite, variation1, variation2, variation3]
    """

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"

    def _hyde(self, query: str) -> str:
        """Generate a HyDE (hypothetical document) passage for the query."""
        prompt = (
            f"Write a short, dense academic passage (3-5 sentences) that directly "
            f"answers the following research question. Do not add a preamble.\n\n"
            f"Question: {query}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"[ERROR] HyDE generation failed: {e} — using original query as fallback")
            return query

    def _expand(self, query: str) -> list[str]:
        """Generate 2 distinct query variations."""
        prompt = (
            f"Generate exactly 2 different search queries for retrieving research papers "
            f"about the following topic. Each query should approach the topic from a different angle. "
            f"Output only the 2 queries, one per line, no numbering or extra text.\n\n"
            f"Topic: {query}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=100,
            )
            lines = [l.strip() for l in response.choices[0].message.content.strip().splitlines() if l.strip()]
            return lines[:2]
        except Exception as e:
            print(f"[ERROR] Query expansion failed: {e} — returning empty variations")
            return []

    def _resolve_with_history(self, query: str, chat_history: list) -> str:
        """
        Rewrite a follow-up query into a self-contained question using recent history.
        Example: history=["Compare BERT and GPT"], query="how is GAN different from them"
                 → "How is GAN different from BERT and GPT?"
        Uses last 3 turns (context window limit). Returns original query on failure.
        """
        # Use last 3 turns to keep context focused and avoid noise
        recent = chat_history[-3:]
        history_text = "\n".join(
            f"User: {t.get('query', '')}\nAssistant: {t.get('answer', '')[:300]}"
            for t in recent
        )
        prompt = (
            f"Given the conversation history below, rewrite the follow-up question "
            f"into a fully self-contained question that resolves all pronouns and "
            f"references (e.g. 'it', 'them', 'that model', 'the first one') so the "
            f"question is clear without the history. "
            f"If the question is already self-contained, return it unchanged. "
            f"Return ONLY the rewritten question, nothing else.\n\n"
            f"Conversation history:\n{history_text}\n\n"
            f"Follow-up question: {query}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=120,
            )
            resolved = response.choices[0].message.content.strip()
            if not resolved:
                return query
            print(f"[QUERY_REWRITE]")
            print(f"[QUERY_REWRITE] original='{query}'")
            print(f"[QUERY_REWRITE] rewritten='{resolved}'")
            return resolved
        except Exception as e:
            print(f"[QUERY_REWRITE] rewrite failed ({e}) — using original query")
            return query

    def transform(self, query: str, chat_history: list = None) -> list[str]:
        """
        Returns all query variants for multi-query retrieval:
          [resolved, HyDE rewrite, variation1, variation2]
        If chat_history is provided, the query is first resolved into a
        standalone self-contained question (pronouns expanded) before expansion.
        index 0 is always the resolved (or original if no history) query.
        HyDE and expand run in parallel to reduce wall-clock latency.
        """
        resolved = (
            self._resolve_with_history(query, chat_history)
            if chat_history
            else query
        )
        # Run HyDE and expand concurrently — both depend only on resolved query
        with ThreadPoolExecutor(max_workers=2) as pool:
            hyde_future = pool.submit(self._hyde, resolved)
            expand_future = pool.submit(self._expand, resolved)
            hyde = hyde_future.result()
            variations = expand_future.result()
        # Cap at 3 — executor enforces the same limit, but apply it here
        # so the logged count is accurate before retrieval starts.
        queries = ([resolved, hyde] + variations)[:3]
        print(f"[MULTI-QUERY] generated queries: {len(queries)} (1 resolved + 1 HyDE + {len(variations)} variations, capped at 3)")
        return queries

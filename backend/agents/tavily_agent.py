import os
import requests
from dotenv import load_dotenv

load_dotenv()

class TavilyAgent:
    def __init__(self):
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing TAVILY_API_KEY in .env")
        self.api_key = api_key

    def search(self, query: str, max_results: int = 25):
        """
        Only return results that have a PDF URL.
        Uses Tavily search + PDF filtering.
        """
        url = "https://api.tavily.com/search"

        # Force Tavily to look for PDF results
        pdf_query = f"{query} filetype:pdf"

        payload = {
            "api_key": self.api_key,
            "query": pdf_query,
            "max_results": max_results,
            "search_type": "advanced",
        }

        try:
            res = requests.post(url, json=payload, timeout=15).json()
        except Exception as e:
            print("❌ Tavily error:", e)
            return []

        pdf_results = []
        for r in res.get("results", []):
            link = r.get("url", "")
            if link.lower().endswith(".pdf"):
                pdf_results.append({
                    "title": r.get("title", "Untitled PDF"),
                    "abstract": r.get("snippet", ""),
                    "summary": r.get("snippet", ""),
                    "link": link,
                    "authors": [],
                    "published": "",
                })

        print(f"📄 Tavily returned {len(pdf_results)} PDF results.")
        return pdf_results

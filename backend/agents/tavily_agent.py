# ==========================================
# tavily_agent.py
# Enhanced Tavily Research Paper Fetcher (5 Papers Version)
# ==========================================

import os
import re
import json
import requests
from dotenv import load_dotenv

class TavilyAgent:
    """
    Tavily-powered agent to search for academic papers and extract valid or inferred PDF links.
    Now fetches up to 5 papers per query, including arXiv, Springer, IEEE, and ACM sources.
    """

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError("❌ Missing TAVILY_API_KEY in environment variables.")
        self.base_url = "https://api.tavily.com/search"
        print("✅ Tavily client initialized successfully")

    # ====================================================
    # 🔍 Search academic papers (max 5)
    # ====================================================
    def search(self, query: str, max_results: int = 5, days: int = 90):
        """
        Searches Tavily for up to 5 academic research papers related to the query.
        Returns a list of papers with titles, abstracts, and direct/fixed PDF links.
        """
        print(f"🔍 Searching Tavily for: {query}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "query": query + " research paper site:arxiv.org OR site:springer.com OR site:ieeexplore.ieee.org OR site:dl.acm.org filetype:pdf",
            "num_results": max_results * 2,  # fetch extra to filter invalid ones
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=40)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"❌ Tavily API error: {e}")
            return []

        results = data.get("results", [])
        if not results:
            print("⚠️ No Tavily results found.")
            return []

        papers = []
        seen = set()

        for r in results:
            url = r.get("url", "")
            title = r.get("title", "Untitled Paper").strip()
            abstract = r.get("snippet", "").strip()

            if not url or url in seen:
                continue
            seen.add(url)

            fixed_link = self._normalize_pdf_url(url)
            if not fixed_link:
                continue

            papers.append({
                "title": title,
                "abstract": abstract,
                "summary": abstract,
                "link": fixed_link,
                "authors": [],
                "published": "",
            })

            if len(papers) >= max_results:
                break

        print(f"📄 Tavily returned {len(papers)} paper results with PDFs.")
        return papers

    # ====================================================
    # 🧩 Normalize academic links into direct PDF links
    # ====================================================
    def _normalize_pdf_url(self, url: str) -> str | None:
        """
        Converts academic URLs into direct or inferred PDF links.
        Handles Springer, IEEE, ACM, arXiv, and other research domains.
        """
        if not url:
            return None

        # ✅ arXiv
        if "arxiv.org/abs/" in url:
            pdf_url = url.replace("abs", "pdf")
            if not pdf_url.endswith(".pdf"):
                pdf_url += ".pdf"
            return pdf_url

        # ✅ Springer
        if "springer" in url:
            match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", url, re.I)
            if match:
                doi = match.group(0)
                return f"https://link.springer.com/content/pdf/{doi}.pdf"

        # ✅ IEEE
        if "ieeexplore.ieee.org/document/" in url:
            doc_id = re.findall(r"/document/(\d+)", url)
            if doc_id:
                return f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={doc_id[0]}"

        # ✅ ACM
        if "dl.acm.org/doi/" in url:
            return url.replace("/doi/", "/doi/pdf/")

        # ✅ Direct PDF links
        if url.endswith(".pdf"):
            return url

        return None


# ====================================================
# 🧪 Example Run (Standalone)
# ====================================================
if __name__ == "__main__":
    agent = TavilyAgent()
    query = input("🧠 Enter your research query: ")
    papers = agent.search(query, max_results=5)
    print(json.dumps(papers, indent=2, ensure_ascii=False))

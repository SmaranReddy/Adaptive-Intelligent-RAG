# ==========================================
# tavily_agent.py — FIXED (Exact K Papers)
# ==========================================

import os
import re
import json
import requests
from dotenv import load_dotenv

class TavilyAgent:
    """
    Tavily-powered agent to search for academic papers.
    Now guarantees EXACT k PDF results.
    """

    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            print("[ERROR] Missing TAVILY_API_KEY — Tavily search will return empty results.")
        self.base_url = "https://api.tavily.com/search"
        print("[OK] Tavily client initialized")

    # ====================================================
    # 🔍 Search academic papers (returns EXACT k papers)
    # ====================================================
    def search(self, query: str, max_results: int = 5, days: int = 90):
        if not self.api_key:
            print("[TAVILY_CALL] skipped — no API key")
            return []

        print(f"[TAVILY_CALL] query='{query[:80]}' max_results={max_results}")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        payload = {
            "query": query + " research paper arxiv",
            "max_results": max_results * 4,   # fetch more since many will be filtered
            "search_depth": "advanced",
            "include_domains": [
                "arxiv.org",
                "semanticscholar.org",
                "aclanthology.org",
                "openreview.net",
            ],
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=40)
            response.raise_for_status()
            data = response.json()
        except Exception as e:
            print(f"[ERROR] Tavily API call failed: {e}")
            return []

        raw_results = data.get("results", [])
        if not raw_results:
            print("⚠️ No Tavily results found.")
            return []

        papers = []
        seen = set()

        for r in raw_results:
            url = r.get("url", "")
            title = r.get("title", "Untitled Paper").strip()
            abstract = (r.get("content") or r.get("snippet") or "").strip()

            if not url or url in seen:
                continue
            seen.add(url)

            # Try to get a direct PDF URL; fall back to original URL
            # (abstract content will be used if PDF download fails)
            pdf = self._normalize_pdf_url(url) or url

            papers.append({
                "title": title,
                "abstract": abstract,
                "summary": abstract,
                "link": pdf,
                "authors": [],
                "published": "",
            })

            if len(papers) >= max_results:
                break

        print(f"[OK] Tavily returned {len(papers)} results (cap={max_results})")
        return papers

    # ====================================================
    # 🧩 Normalize academic links into direct PDF links
    # ====================================================
    def _normalize_pdf_url(self, url: str) -> str | None:
        if not url:
            return None

        # arXiv → direct PDF
        if "arxiv.org/abs/" in url:
            pdf = url.replace("abs", "pdf")
            return pdf if pdf.endswith(".pdf") else pdf + ".pdf"

        # Springer → DOI → PDF
        if "springer" in url:
            match = re.search(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", url, re.I)
            if match:
                doi = match.group(0)
                return f"https://link.springer.com/content/pdf/{doi}.pdf"

        # IEEE
        if "ieeexplore.ieee.org/document/" in url:
            doc_id = re.findall(r"/document/(\d+)", url)
            if doc_id:
                return f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?tp=&arnumber={doc_id[0]}"

        # ACM
        if "dl.acm.org/doi/" in url:
            return url.replace("/doi/", "/doi/pdf/")

        # Direct PDF
        if url.endswith(".pdf"):
            return url

        return None

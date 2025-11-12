# ==========================================
# tavily_agent.py
# Smart PDF Paper Retriever using Tavily API (Class Version + search() wrapper)
# ==========================================

from tavily import TavilyClient
from dotenv import load_dotenv
import os
import re
import json


class TavilyAgent:
    """
    Tavily-powered agent to search academic papers and extract direct or inferred PDF links.
    Includes a compatibility .search() wrapper for legacy pipeline calls.
    """

    def __init__(self):
        load_dotenv()
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            raise ValueError("❌ Missing TAVILY_API_KEY in environment variables.")
        self.client = TavilyClient(api_key=api_key)
        print("✅ Tavily client initialized successfully")

    # ====================================================
    # 🧩 Normalize URLs into valid direct PDF download links
    # ====================================================
    @staticmethod
    def normalize_to_pdf(url: str) -> str | None:
        """Cleans and converts academic URLs to direct PDF links where possible."""
        if not url:
            return None

        # arXiv
        if "arxiv.org/abs/" in url:
            url = re.sub(r"arxiv\.org/abs/", "arxiv.org/pdf/", url)
            if not url.endswith(".pdf"):
                url += ".pdf"
            return url
        if "arxiv.org/pdf/" in url and not url.endswith(".pdf"):
            return url + ".pdf"

        # IEEE
        if "ieeexplore.ieee.org" in url:
            match = re.search(r"arnumber=(\d+)", url)
            if match:
                arnum = match.group(1)
                return f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?arnumber={arnum}"

        # ACM
        if "dl.acm.org" in url:
            return url.replace("/doi/", "/doi/pdf/")

        # Springer
        if "springer.com" in url and not url.endswith(".pdf"):
            return url.replace("chapter/", "content/pdf/") + ".pdf"

        # ResearchGate
        if "researchgate.net" in url:
            return url

        # Direct .pdf link
        if url.endswith(".pdf"):
            return url

        return None

    # ====================================================
    # 🧩 Extract and clean all PDF links
    # ====================================================
    def extract_pdf_links(self, results: list[dict]) -> list[str]:
        """Extracts and normalizes PDF URLs from Tavily search results."""
        pdf_links = []
        for r in results:
            url = r.get("url", "")
            pdf_url = self.normalize_to_pdf(url)
            if pdf_url:
                pdf_links.append(pdf_url)

        # Deduplicate & validate
        cleaned = []
        for link in dict.fromkeys(pdf_links):  # preserves order
            if re.match(r"^https?://", link):
                cleaned.append(link)
        return cleaned

    # ====================================================
    # 🔍 Main: Fetch best PDF link for a given research query
    # ====================================================
    def fetch_paper_pdf(self, query: str, days: int = 90, max_results: int = 10) -> str | None:
        """
        Search for research papers and return the most relevant PDF link (if available).
        Also saves results to 'paper_result.json'.
        """
        print(f"\n🔍 Searching for papers related to: '{query}' ...")

        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                topic="general",
                include_answer=False,
                include_domains=[
                    "arxiv.org",
                    "researchgate.net",
                    "dl.acm.org",
                    "ieeexplore.ieee.org",
                    "springer.com",
                ],
                include_images=False,
                include_raw_content=False,
                max_results=max_results,
                days=days,
            )
        except Exception as e:
            print(f"❌ Tavily API error: {e}")
            return None

        results = response.get("results", [])
        if not results:
            print("❌ No results found.")
            return None

        pdf_links = self.extract_pdf_links(results)
        if not pdf_links:
            print("⚠ No direct or inferred PDF links found.")
            return None

        best_link = pdf_links[0]
        print(f"\n✅ Top Paper PDF Link Found:\n{best_link}\n")

        # Save to JSON
        output = {
            "query": query,
            "pdf_link": best_link,
            "all_found_pdfs": pdf_links,
        }

        with open("paper_result.json", "w", encoding="utf-8") as f:
            json.dump(output, f, indent=4, ensure_ascii=False)

        print("📄 Results saved to paper_result.json")
        return best_link

    # ====================================================
    # 🌐 Compatibility Wrapper for pipeline nodes
    # ====================================================
    def search(self, query: str, max_results: int = 10, days: int = 90) -> list[dict]:
        """
        Compatibility wrapper for pipeline nodes using tavily.search().
        Returns a list of papers (dict) with metadata and PDF links.
        """
        print(f"🔍 Searching Tavily for: {query}")

        try:
            response = self.client.search(
                query=query,
                search_depth="advanced",
                topic="general",
                include_answer=False,
                include_domains=[
                    "arxiv.org",
                    "researchgate.net",
                    "dl.acm.org",
                    "ieeexplore.ieee.org",
                    "springer.com",
                ],
                include_images=False,
                include_raw_content=False,
                max_results=max_results,
                days=days,
            )
        except Exception as e:
            print(f"❌ Tavily API error: {e}")
            return []

        results = response.get("results", [])
        if not results:
            print("❌ No results found.")
            return []

        pdf_links = self.extract_pdf_links(results)
        papers = []
        for r in results:
            link = self.normalize_to_pdf(r.get("url", ""))
            if link and link in pdf_links:
                papers.append({
                    "title": r.get("title", "Untitled PDF").strip(),
                    "abstract": r.get("snippet", "").strip(),
                    "summary": r.get("snippet", "").strip(),
                    "link": link,
                    "authors": [],
                    "published": "",
                })

        print(f"📄 Tavily returned {len(papers)} paper results with PDFs.")
        return papers


# ====================================================
# 🧪 Example Run (Standalone)
# ====================================================
if __name__ == "__main__":
    agent = TavilyAgent()
    query = input("🧠 Enter your research query: ")
    agent.fetch_paper_pdf(query)

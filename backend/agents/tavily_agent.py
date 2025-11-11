# ==========================================
# tavily-agent.py
# Smart PDF Paper Retriever using Tavily API
# ==========================================

from tavily import TavilyClient
from dotenv import load_dotenv
import os
import re
import json
import requests

# ======================================
# 1️⃣ Load Environment and Initialize API
# ======================================
load_dotenv()

api_key = os.getenv("TAVILY_API_KEY")
if not api_key:
    raise ValueError("❌ Missing TAVILY_API_KEY in environment variables.")

tavily_client = TavilyClient(api_key=api_key)


# ======================================
# 2️⃣ Helper Function: Clean + Fix PDF Links
# ======================================
def normalize_to_pdf(url: str) -> str | None:
    """
    Cleans up and converts known academic URLs to direct PDF links where possible.
    Returns None if no valid PDF path can be inferred.
    """
    if not url:
        return None

    # 🧩 arXiv fix: abs → pdf
    if "arxiv.org/abs/" in url:
        url = re.sub(r"arxiv\.org/abs/", "arxiv.org/pdf/", url)
        if not url.endswith(".pdf"):
            url += ".pdf"
        return url

    # 🧩 arXiv already a PDF
    if "arxiv.org/pdf/" in url and not url.endswith(".pdf"):
        return url + ".pdf"

    # 🧩 IEEE / ACM / Springer known patterns
    if "ieeexplore.ieee.org" in url:
        # IEEE often hosts PDFs under /stampPDF/getPDF.jsp?tp=&arnumber=XXXXXXX
        match = re.search(r"arnumber=(\d+)", url)
        if match:
            arnum = match.group(1)
            return f"https://ieeexplore.ieee.org/stampPDF/getPDF.jsp?arnumber={arnum}"

    if "dl.acm.org" in url:
        # ACM PDFs are often found by replacing "doi/" with "doi/pdf/"
        return url.replace("/doi/", "/doi/pdf/")

    if "springer.com" in url and not url.endswith(".pdf"):
        # Springer PDFs often have 'content/pdf' in link — guess pattern
        return url.replace("chapter/", "content/pdf/") + ".pdf"

    # 🧩 Direct PDF link
    if url.endswith(".pdf"):
        return url

    # 🧩 ResearchGate — link may require login, keep for manual review
    if "researchgate.net" in url:
        return url

    # ❌ Non-pdf, skip
    return None


# ======================================
# 3️⃣ Extract and Clean All PDF Links
# ======================================
def extract_pdf_links(results):
    """
    Extracts and normalizes PDF URLs from Tavily results.
    """
    pdf_links = []
    for r in results:
        url = r.get("url", "")
        pdf_url = normalize_to_pdf(url)
        if pdf_url:
            pdf_links.append(pdf_url)

    # Deduplicate & filter clearly invalid ones
    cleaned = []
    for link in dict.fromkeys(pdf_links):  # deduplicate
        if re.match(r"^https?://", link):
            cleaned.append(link)
    return cleaned


# ======================================
# 4️⃣ Main Function: Fetch Paper PDF
# ======================================
def fetch_paper_pdf(query: str, days: int = 90, max_results: int = 10):
    """
    Search for research papers and return the most relevant PDF link (if available).
    """

    print(f"\n🔍 Searching for papers related to: '{query}' ...")

    response = tavily_client.search(
        query=query,
        search_depth="advanced",
        topic="general",
        include_answer=False,
        include_domains=[
            "arxiv.org", "researchgate.net", "dl.acm.org", "ieeexplore.ieee.org", "springer.com"
        ],
        include_images=False,
        include_raw_content=False,
        max_results=max_results,
        days=days,
    )

    results = response.get("results", [])
    if not results:
        print("❌ No results found.")
        return None

    pdf_links = extract_pdf_links(results)
    if not pdf_links:
        print("⚠️ No direct or inferred PDF links found.")
        return None

    best_link = pdf_links[0]
    print(f"\n✅ Top Paper PDF Link Found:\n{best_link}\n")

    # Optional: save output
    output = {
        "query": query,
        "pdf_link": best_link,
        "all_found_pdfs": pdf_links,
    }

    with open("paper_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)

    print("📄 Results saved to paper_result.json")

    return best_link


# ======================================
# 5️⃣ Example Run (Standalone)
# ======================================
if __name__ == "__main__":
    query = input("🧠 Enter your research query: ")
    fetch_paper_pdf(query)
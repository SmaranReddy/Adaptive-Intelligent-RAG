import feedparser
import requests
import fitz  # PyMuPDF
import os

# 1. Query ArXiv
query = "Attention is all you need"
max_results = 3
url = f"https://export.arxiv.org/api/query?search_query=all:{query.replace(' ', '+')}&start=0&max_results={max_results}"

feed = feedparser.parse(url)

papers = []

for entry in feed.entries:
    paper_data = {
        "title": entry.title,
        "authors": [author.name for author in entry.authors],
        "published": entry.published,
        "abstract": entry.summary,
        "link": entry.link
    }

    print(f"Title: {paper_data['title']}")
    print(f"Authors: {paper_data['authors']}")
    print(f"Published: {paper_data['published']}")
    print(f"Abstract: {paper_data['abstract'][:200]} ...")
    print(f"Link: {paper_data['link']}")

    # 2. Download PDF
    pdf_url = entry.link.replace("abs", "pdf")
    pdf_filename = f"{paper_data['title'].replace('/', '_')}.pdf"
    try:
        response = requests.get(pdf_url)
        with open(pdf_filename, "wb") as f:
            f.write(response.content)
        print(f"Downloaded PDF: {pdf_filename}")

        # 3. Extract full text
        doc = fitz.open(pdf_filename)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        paper_data["full_text"] = full_text
        print(f"Extracted {len(full_text)} characters of text")
    except Exception as e:
        print(f"Failed to download/extract PDF for {paper_data['title']}: {e}")
        paper_data["full_text"] = None

    print("-" * 80)
    papers.append(paper_data)

# Optional: inspect first 1000 chars of first paper
if papers and papers[0]["full_text"]:
    print("\n--- Sample Full Text ---\n")
    print(papers[0]["full_text"][:1000])

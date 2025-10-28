import requests
import fitz  # PyMuPDF
import os

class ParserAgent:
    def __init__(self, download_dir="pdfs"):
        self.download_dir = download_dir
        os.makedirs(download_dir, exist_ok=True)

    def parse_papers(self, papers):
        """
        Receives a list of papers from RetrieverAgent.
        Downloads PDFs and extracts full text.
        Returns papers with 'full_text' field added.
        """
        for paper in papers:
            title_safe = paper['title'].replace("/", "_").replace(":", "_")
            pdf_filename = os.path.join(self.download_dir, f"{title_safe}.pdf")

            # Convert abstract URL to PDF URL
            pdf_url = paper.get("link", "").replace("abs", "pdf") + ".pdf"

            try:
                print(f"Downloading PDF for: {paper['title']}")
                response = requests.get(pdf_url)
                response.raise_for_status()

                with open(pdf_filename, "wb") as f:
                    f.write(response.content)

                # Extract full text
                doc = fitz.open(pdf_filename)
                full_text = "".join([page.get_text() for page in doc])
                paper["full_text"] = full_text
                print(f"Extracted {len(full_text)} characters for: {paper['title']}")

            except Exception as e:
                print(f"Failed to process {paper['title']}: {e}")
                paper["full_text"] = None

        return papers

# backend/agents/downloader_agent.py
import requests
import fitz  # PyMuPDF
import os
import re

class DownloaderAgent:
    def __init__(self, save_dir="downloads"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def _sanitize_filename(self, title: str) -> str:
        """Remove invalid characters and normalize spaces."""
        safe = re.sub(r'[\\/*?:"<>|]', "", title)
        safe = safe.replace("\n", " ").replace("\r", " ").strip()
        return " ".join(safe.split())

    def download_and_extract(self, paper: dict) -> dict:
        """Download the paper PDF and extract text safely."""
        pdf_url = paper["link"].replace("abs", "pdf")
        safe_title = self._sanitize_filename(paper["title"])
        pdf_path = os.path.join(self.save_dir, f"{safe_title}.pdf")

        try:
            # 1. Download PDF
            response = requests.get(pdf_url, timeout=15)
            response.raise_for_status()
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            print(f"📥 Downloaded: {pdf_path}")

            # 2. Extract text and ensure file is closed
            full_text = []
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    full_text.append(page.get_text())
            paper["full_text"] = "\n".join(full_text)

            print(f"📄 Extracted {len(paper['full_text'])} characters.")
        except Exception as e:
            print(f"❌ Failed to process {safe_title}: {e}")
            paper["full_text"] = None

        return paper

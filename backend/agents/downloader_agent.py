import requests
import PyPDF2
from io import BytesIO
import os
import re
import json
from time import sleep

BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
}

class DownloaderAgent:
    def __init__(self, save_dir="downloads"):
        os.makedirs(save_dir, exist_ok=True)
        self.session = requests.Session()
        self.session.headers.update(BROWSER_HEADERS)
        self.save_dir = save_dir

    def _sanitize_filename(self, title: str) -> str:
        safe = re.sub(r'[\\/*?:"<>|]', "", title or "untitled")
        return " ".join(safe.split())

    def download_and_extract(self, item: dict) -> dict:
        url = item.get("link", "")
        title = item.get("title", "untitled")
        safe_title = self._sanitize_filename(title)
        pdf_path = os.path.join(self.save_dir, safe_title + ".pdf")
        json_path = os.path.join(self.save_dir, safe_title + ".json")

        if not url:
            print("⚠️ No URL detected")
            item["full_text"] = ""
            return item

        try:
            pdf_bytes = self._attempt_pdf_download(url)
            if not pdf_bytes:
                print(f"❌ Could not download PDF: {url}")
                item["full_text"] = ""
                return item

            # Save PDF
            with open(pdf_path, "wb") as f:
                f.write(pdf_bytes)
            print(f"📄 Saved PDF: {pdf_path}")

            # Extract text
            full_text = self._extract_text(pdf_bytes)
            item["full_text"] = full_text

            # Save metadata
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump({
                    "title": item.get("title"),
                    "url": url,
                    "chars": len(full_text),
                    "preview": full_text[:1500],
                }, f, ensure_ascii=False, indent=2)
            print(f"💾 Saved JSON: {json_path}")

        except Exception as e:
            print(f"❌ PDF extraction failed: {e}")
            item["full_text"] = ""

        return item

    # ------------------------------------------
    # ✅ Strong PDF downloader with retries
    # ------------------------------------------
    def _attempt_pdf_download(self, url):
        try_urls = [url]

        # DigitalCommons / ResearchGate trick
        if "viewcontent" in url:
            # They expect direct access
            try_urls.append(url.replace("viewcontent.cgi", "download"))

        for attempt in range(3):
            for u in try_urls:
                try:
                    resp = self.session.get(u, timeout=20, allow_redirects=True)
                    if resp.status_code == 403:
                        print(f"⚠️ 403 Forbidden, retrying with referer spoof...")
                        self.session.headers["Referer"] = u
                        sleep(1)
                        continue

                    # Check PDF content type
                    if "application/pdf" in resp.headers.get("Content-Type", "").lower():
                        return resp.content

                    # Some servers don't send correct content-type, but PDF bytes still readable
                    if resp.content[:4] == b"%PDF":
                        return resp.content

                except Exception as e:
                    print(f"⚠️ Retry failed ({e})")

            sleep(1)  # small delay between attempts

        return None

    # ------------------------------------------
    # ✅ PDF text extractor (safe)
    # ------------------------------------------
    def _extract_text(self, pdf_bytes):
        try:
            reader = PyPDF2.PdfReader(BytesIO(pdf_bytes))
            text = []
            for page in reader.pages:
                t = page.extract_text() or ""
                text.append(t)
            return "\n".join(text).strip()
        except Exception as e:
            print(f"⚠️ PDF parsing error: {e}")
            return ""

import requests
import xml.etree.ElementTree as ET

class RetrieverAgent:
    def search_papers(self, query, max_results=3):
        """
        Searches arXiv for papers matching `query`.
        Returns a list of dicts: title, abstract, and PDF link.
        """
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
        response = requests.get(url)
        response.raise_for_status()  # Raise error if request fails

        # Parse XML
        root = ET.fromstring(response.text)
        papers = []

        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title_elem = entry.find("{http://www.w3.org/2005/Atom}title")
            abstract_elem = entry.find("{http://www.w3.org/2005/Atom}summary")
            link_elem = entry.find("{http://www.w3.org/2005/Atom}id")  # arXiv abstract URL

            if title_elem is not None and abstract_elem is not None and link_elem is not None:
                title = title_elem.text.strip()
                abstract = abstract_elem.text.strip()
                link = link_elem.text.strip()

                papers.append({
                    "title": title,
                    "abstract": abstract,
                    "link": link
                })

        return papers

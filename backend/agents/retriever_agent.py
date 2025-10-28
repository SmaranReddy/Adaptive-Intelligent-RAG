# backend/agents/retriever_agent.py
import requests
import xml.etree.ElementTree as ET

class RetrieverAgent:
    # --- THIS LINE WAS CHANGED ---
    def search_papers(self, query, max_results=10):
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
        response = requests.get(url)

        # Parse XML properly
        root = ET.fromstring(response.text)
        papers = []
        
        # Define the Atom namespace to find elements correctly
        namespaces = {'atom': 'http://www.w3.org/2005/Atom'}

        for entry in root.findall("atom:entry", namespaces):
            title_elem = entry.find("atom:title", namespaces)
            abstract_elem = entry.find("atom:summary", namespaces)

            if title_elem is not None and abstract_elem is not None:
                title = title_elem.text.strip()
                abstract = abstract_elem.text.strip()
                papers.append({
                    "title": title,
                    "abstract": abstract
                })

        return papers

# backend/agents/arxiv_agent.py
import feedparser

class ArxivAgent:
    def fetch_papers(self, query, max_results=3):
        url = f"https://export.arxiv.org/api/query?search_query=all:{query.replace(' ', '+')}&start=0&max_results={max_results}"
        feed = feedparser.parse(url)

        papers = []
        for entry in feed.entries:
            papers.append({
                "title": entry.title,
                "authors": [author.name for author in entry.authors],
                "published": entry.published,
                "abstract": entry.summary,
                "link": entry.link
            })
        return papers

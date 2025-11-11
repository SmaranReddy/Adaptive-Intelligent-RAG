# backend/graph_state.py
from typing import List, Dict, Optional, TypedDict
from typing_extensions import NotRequired

class Paper(TypedDict, total=False):
    title: str
    link: str
    abstract: str
    authors: List[str]
    published: str
    summary_structured: NotRequired[str]
    full_text: NotRequired[str]
    safe_title: NotRequired[str]
    chunks: NotRequired[List[str]]
    chunk_embeddings: NotRequired[List[List[float]]]
    upsert_responses: NotRequired[List[Dict]]

class GraphState(TypedDict, total=False):
    user_query: str
    num_papers: int
    papers: List[Paper]
    cleaned_texts: List[str]
    token_counts: List[int]
    errors: List[str]
    final_message: Optional[str]

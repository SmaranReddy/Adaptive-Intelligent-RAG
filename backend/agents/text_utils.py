# backend/agents/text_utils.py
import re
import tiktoken

class PreprocessingAgent:
    """Cleans raw text extracted from PDF before embedding."""
    def preprocess(self, text: str) -> str:
        if not text:
            return ""
        text = re.sub(r"\n+", "\n", text)
        text = re.split(r"\bReferences\b", text, maxsplit=1)[0]
        text = re.sub(r"Figure\s*\d+|Table\s*\d+", "", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text


class TokenizerAgent:
    """Tokenizes text to count or truncate tokens."""
    def __init__(self, model="text-embedding-3-small"):
        try:
            self.enc = tiktoken.encoding_for_model(model)
        except:
            self.enc = None

    def count_tokens(self, text: str) -> int:
        if self.enc:
            return len(self.enc.encode(text))
        return len(text.split())

    def truncate(self, text: str, max_tokens: int = 8000) -> str:
        if self.enc:
            tokens = self.enc.encode(text)
            return self.enc.decode(tokens[:max_tokens])
        return " ".join(text.split()[:max_tokens])


class ChunkingAgent:
    """Splits text into overlapping chunks for embedding."""
    def __init__(self, chunk_size: int = 1500, overlap: int = 200):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk_text(self, text: str):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+self.chunk_size])
            start += self.chunk_size - self.overlap
        return chunks

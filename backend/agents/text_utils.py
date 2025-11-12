import re
import nltk
from nltk.tokenize import sent_tokenize

# Ensure required tokenizers are downloaded
for pkg in ["punkt", "punkt_tab"]:
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg)

# Download punkt tokenizer (if not already installed)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

# -------------------------------
# 1️⃣ Text Preprocessing
# -------------------------------

def preprocess_text(text: str) -> str:
    """
    Cleans extracted text by removing unnecessary symbols, multiple spaces,
    and other noise for better embedding quality.
    """
    if not text:
        return ""

    # Remove non-printable / control characters
    text = re.sub(r"[^\x20-\x7E\n]", " ", text)

    # Replace multiple newlines with one
    text = re.sub(r"\n+", "\n", text)

    # Replace multiple spaces/tabs with one
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive hyphenations and page breaks
    text = re.sub(r"-\s*\n", "", text)
    text = text.replace("\r", "")

    # Clean up unwanted artifacts (URLs, figure/table tags)
    text = re.sub(r"(https?://\S+)", "", text)
    text = re.sub(r"(FIGURE|TABLE|REFERENCES|DOI:)\s*\d*", "", text, flags=re.IGNORECASE)

    # Trim leading/trailing spaces
    text = text.strip()

    return text


# -------------------------------
# 2️⃣ Text Chunking
# -------------------------------

def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 150) -> list:
    """
    Splits text into overlapping chunks for embedding and retrieval.
    """
    if not text:
        return []

    # Split into sentences for cleaner chunking
    sentences = sent_tokenize(text)

    chunks = []
    current_chunk = ""

    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += " " + sentence
        else:
            chunks.append(current_chunk.strip())
            # Keep an overlap for context
            current_chunk = sentence[-overlap:] if overlap < len(sentence) else sentence

    if current_chunk.strip():
        chunks.append(current_chunk.strip())

    return chunks

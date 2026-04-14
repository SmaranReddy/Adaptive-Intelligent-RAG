from fastembed import TextEmbedding

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384

_model: TextEmbedding = None


def get_embedding_model() -> TextEmbedding:
    global _model
    if _model is None:
        _model = TextEmbedding(EMBED_MODEL)
        print(f"[OK] Embedding model ({EMBED_MODEL}, dim={EMBED_DIM}) loaded.")
    return _model

# Convert text into numeric vectors with a sentence-transformers model.

from sentence_transformers import SentenceTransformer

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class Embedder:
    # Small wrapper around a sentence-transformers embedding model

    def __init__(self, model_name=DEFAULT_MODEL):
        self.model_name = model_name
        # The model runs on CPU, which is enough for small documents
        self.model = SentenceTransformer(model_name, device="cpu")

    def encode(self, texts):
        # Return a normalised float32 matrix with one row per text
        if not texts:
            raise ValueError("No text was given to embed.")
        vectors = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.astype("float32")

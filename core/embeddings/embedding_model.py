from sentence_transformers import SentenceTransformer
import numpy as np
from app.config import EMBEDDING_MODEL


class EmbeddingModel:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL)

    def embed(self, texts):
        """
        Returns L2-normalized float32 embeddings
        optimized for FAISS similarity search
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True  # 🔥 important
        )

        return np.array(embeddings).astype("float32")

import numpy as np
from core.embeddings.embedding_model import EmbeddingModel
from vectorstore.faiss_store import FAISSStore


class DenseRetriever:

    def __init__(self):
        self.embedder = EmbeddingModel()
        self.index, self.metadata = FAISSStore.load()

    def retrieve(self, query, top_k=5):

        # Generate query embedding
        query_embedding = self.embedder.embed([query])

        # Search FAISS index
        distances, indices = self.index.search(
            np.array(query_embedding).astype("float32"),
            top_k
        )

        results = []

        for idx, dist in zip(indices[0], distances[0]):

            # Convert L2 distance → similarity score
            similarity = 1 / (1 + float(dist))

            results.append({
                "text": self.metadata[idx]["text"],
                "source": self.metadata[idx]["source"],
                "score": float(dist),              # raw FAISS distance
                "dense_similarity": similarity     # normalized similarity
            })

        return results

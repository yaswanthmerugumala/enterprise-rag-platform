from core.retrieval.dense_retriever import DenseRetriever
from core.retrieval.sparse_retriever import SparseRetriever


class HybridRetriever:

    def __init__(self, alpha=0.7):
        """
        alpha → weight for dense retrieval
        (1 - alpha) → weight for sparse retrieval
        """
        self.dense = DenseRetriever()
        self.sparse = SparseRetriever()
        self.alpha = alpha

    def retrieve(self, query, top_k=5):

        # Get more candidates before fusion
        dense_results = self.dense.retrieve(query, top_k=top_k * 2)
        sparse_results = self.sparse.retrieve(query, top_k=top_k * 2)

        combined = {}

        # ---- Normalize Sparse Scores ----
        sparse_scores = [r["score"] for r in sparse_results]
        max_sparse = max(sparse_scores) if sparse_scores else 1

        # ---- Process Dense Results ----
        for r in dense_results:

            # Convert L2 distance to similarity
            dense_similarity = 1 / (1 + r.get("score", 1))

            combined[r["text"]] = {
                "text": r["text"],
                "source": r["source"],
                "dense_score": dense_similarity,
                "sparse_score": 0.0,
                "hybrid_score": self.alpha * dense_similarity
            }

        # ---- Process Sparse Results ----
        for r in sparse_results:

            normalized_sparse = r["score"] / max_sparse if max_sparse != 0 else 0

            if r["text"] in combined:
                combined[r["text"]]["sparse_score"] = normalized_sparse
                combined[r["text"]]["hybrid_score"] += (
                    (1 - self.alpha) * normalized_sparse
                )
            else:
                combined[r["text"]] = {
                    "text": r["text"],
                    "source": r["source"],
                    "dense_score": 0.0,
                    "sparse_score": normalized_sparse,
                    "hybrid_score": (1 - self.alpha) * normalized_sparse
                }

        # ---- Sort by hybrid score ----
        results = list(combined.values())
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)

        return results[:top_k]

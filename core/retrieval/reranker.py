from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, contexts):

        pairs = [[query, c["text"]] for c in contexts]
        scores = self.model.predict(pairs)

        for i, score in enumerate(scores):
            contexts[i]["rerank_score"] = float(score)

        contexts.sort(key=lambda x: x["rerank_score"], reverse=True)

        return contexts

from core.retrieval.reranker import Reranker
from core.retrieval.hybrid_retriever import HybridRetriever

h = HybridRetriever()
r = Reranker()

retrieved = h.retrieve("encryption standard", top_k=5)
reranked = r.rerank("encryption standard", retrieved)

print(reranked)

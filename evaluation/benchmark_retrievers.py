import json
import time

from core.retrieval.dense_retriever import DenseRetriever
from core.retrieval.hybrid_retriever import HybridRetriever
from core.retrieval.reranker import Reranker
from evaluation.faithfulness import faithfulness_score


# ===========================
# Metrics
# ===========================

def recall_at_k(retrieved_sources, relevant_sources):
    if not relevant_sources:
        return 1 if not retrieved_sources else 0
    return int(any(src in relevant_sources for src in retrieved_sources))


def precision_at_k(retrieved_sources, relevant_sources):
    if not retrieved_sources:
        return 0
    if not relevant_sources:
        return 1 if not retrieved_sources else 0
    correct = sum(1 for src in retrieved_sources if src in relevant_sources)
    return correct / len(retrieved_sources)


def reciprocal_rank(retrieved_sources, relevant_sources):
    for i, src in enumerate(retrieved_sources):
        if src in relevant_sources:
            return 1 / (i + 1)
    return 0


# ===========================
# Evaluation Engine
# ===========================

def evaluate_retriever(name, retrieve_fn, dataset, k=5):

    total_recall = 0
    total_precision = 0
    total_mrr = 0
    total_latency = 0

    print(f"\n===== Evaluating {name} =====")

    for item in dataset:

        query = item["query"]
        relevant = item["relevant_sources"]

        start = time.time()
        results = retrieve_fn(query)
        end = time.time()

        latency = end - start
        total_latency += latency

        # Extract document-level unique sources
        retrieved_sources = list(dict.fromkeys(
            [r["source"] for r in results]
        ))[:k]   # enforce true Recall@K

        r_at_k = recall_at_k(retrieved_sources, relevant)
        p_at_k = precision_at_k(retrieved_sources, relevant)
        rr = reciprocal_rank(retrieved_sources, relevant)

        total_recall += r_at_k
        total_precision += p_at_k
        total_mrr += rr

        print(f"\nQuery: {query}")
        print("Relevant:", relevant)
        print("Retrieved:", retrieved_sources)
        print("Recall@K:", r_at_k)
        print("Precision@K:", round(p_at_k, 3))
        print("Reciprocal Rank:", round(rr, 3))
        print("Latency:", round(latency, 3))

    n = len(dataset)

    return {
        "Recall@K": round(total_recall / n, 3),
        "Precision@K": round(total_precision / n, 3),
        "MRR": round(total_mrr / n, 3),
        "Avg Latency (s)": round(total_latency / n, 3)
    }


# ===========================
# Main
# ===========================

def main():

    with open("evaluation/gold_dataset.json", "r") as f:
        dataset = json.load(f)

    dense = DenseRetriever()
    hybrid = HybridRetriever()
    reranker = Reranker()

    # --- Retrieval strategies ---

    dense_fn = lambda q: dense.retrieve(q, top_k=5)

    hybrid_fn = lambda q: hybrid.retrieve(q, top_k=5)

    hybrid_rerank_fn = lambda q: reranker.rerank(
        q,
        hybrid.retrieve(q, top_k=10)
    )[:5]

    # --- Run benchmarks ---

    dense_metrics = evaluate_retriever("Dense", dense_fn, dataset)
    hybrid_metrics = evaluate_retriever("Hybrid", hybrid_fn, dataset)
    hybrid_rerank_metrics = evaluate_retriever("Hybrid + Rerank", hybrid_rerank_fn, dataset)

    print("\n\n========== FINAL COMPARISON ==========")
    print("Dense:", dense_metrics)
    print("Hybrid:", hybrid_metrics)
    print("Hybrid + Rerank:", hybrid_rerank_metrics)


if __name__ == "__main__":
    main()

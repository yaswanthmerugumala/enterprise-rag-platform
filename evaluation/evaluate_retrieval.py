import json
import time

from core.retrieval.hybrid_retriever import HybridRetriever
from core.llm.ollama_llm import OllamaLLM
from evaluation.faithfulness import faithfulness_score


# ===========================
# Metrics
# ===========================

def recall_at_k(retrieved_sources, relevant_sources):
    return int(any(src in relevant_sources for src in retrieved_sources))


def precision_at_k(retrieved_sources, relevant_sources):
    if not retrieved_sources:
        return 0
    correct = sum(1 for src in retrieved_sources if src in relevant_sources)
    return correct / len(retrieved_sources)


def reciprocal_rank(retrieved_sources, relevant_sources):
    for idx, src in enumerate(retrieved_sources):
        if src in relevant_sources:
            return 1 / (idx + 1)
    return 0


# ===========================
# Evaluation
# ===========================

def evaluate():

    with open("evaluation/gold_dataset.json", "r") as f:
        dataset = json.load(f)

    retriever = HybridRetriever()
    llm = OllamaLLM()

    total_recall = 0
    total_precision = 0
    total_mrr = 0
    total_latency = 0
    total_faithfulness = 0

    for item in dataset:

        query = item["query"]
        relevant = item["relevant_sources"]

        # ⏱️ Measure latency
        start = time.time()
        results = retriever.retrieve(query, top_k=5)
        end = time.time()

        latency = end - start
        total_latency += latency

        # ✅ Deduplicate sources (important fix)
        retrieved_sources = list(dict.fromkeys(
            [r["source"] for r in results]
        ))

        # 📊 Metrics
        r_at_5 = recall_at_k(retrieved_sources, relevant)
        p_at_5 = precision_at_k(retrieved_sources, relevant)
        rr = reciprocal_rank(retrieved_sources, relevant)

        total_recall += r_at_5
        total_precision += p_at_5
        total_mrr += rr

        # ===========================
        # 🔥 Faithfulness
        # ===========================

        context_text = "\n\n".join([r["text"] for r in results])

        prompt = f"""
Answer ONLY from the context.
If unsure, say "I don't know."

Context:
{context_text}

Question:
{query}
"""

        answer = llm.generate(prompt)

        faith = faithfulness_score(query, context_text, answer)
        total_faithfulness += faith

        # ===========================
        # 🖨️ Logs
        # ===========================

        print(f"\nQuery: {query}")
        print("Relevant:", relevant)
        print("Retrieved:", retrieved_sources)
        print("Recall@5:", r_at_5)
        print("Precision@5:", round(p_at_5, 3))
        print("Reciprocal Rank:", round(rr, 3))
        print("Faithfulness:", round(faith, 3))
        print("Latency:", round(latency, 3))

    n = len(dataset)

    print("\n===== FINAL METRICS =====")
    print("Recall@5:", round(total_recall / n, 3))
    print("Precision@5:", round(total_precision / n, 3))
    print("MRR:", round(total_mrr / n, 3))
    print("Faithfulness:", round(total_faithfulness / n, 3))
    print("Avg Latency (s):", round(total_latency / n, 3))


if __name__ == "__main__":
    evaluate()

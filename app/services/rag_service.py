import time
from typing import Generator, Dict, Any
from cachetools import TTLCache
from loguru import logger

from core.retrieval.hybrid_retriever import HybridRetriever
from core.retrieval.reranker import Reranker
from core.llm.ollama_llm import OllamaLLM
from core.guardrails.input_guard import InputGuard
from evaluation.faithfulness import faithfulness_score

# ✅ Import Prometheus metrics
from app.metrics import CACHE_HITS, FAITHFULNESS_SCORE


class RAGService:

    def __init__(self):
        self.retriever = HybridRetriever()
        self.reranker = Reranker()
        self.llm = OllamaLLM()
        self.guard = InputGuard()
        self.cache = TTLCache(maxsize=100, ttl=300)

    # ==========================================================
    # NORMAL RESPONSE
    # ==========================================================
    def answer_query(self, query: str) -> Dict[str, Any]:

        query = query.strip()
        logger.info(f"Query: {query}")

        # ✅ Empty query handling
        if not query:
            return {
                "answer": "Please provide a valid question.",
                "sources": [],
                "latency_seconds": 0,
                "faithfulness": 1.0,
                "cached": False
            }

        # ✅ Cache check
        if query in self.cache:
            CACHE_HITS.inc()  # 🔥 track cache hit

            result = self.cache[query].copy()
            result["cached"] = True

            logger.info("Cache hit")
            return result

        start = time.time()

        # ✅ Guard check
        is_valid, message = self.guard.validate(query)
        if not is_valid:
            logger.warning(f"Blocked query: {query}")

            return {
                "answer": message,
                "sources": [],
                "latency_seconds": 0,
                "faithfulness": 1.0,
                "cached": False
            }

        try:
            # ✅ Retrieval
            retrieved = self.retriever.retrieve(query, top_k=10)

            if not retrieved:
                logger.warning("No documents retrieved")

                return {
                    "answer": "No relevant information found.",
                    "sources": [],
                    "latency_seconds": 0,
                    "faithfulness": 0.0,
                    "cached": False
                }

            # ✅ Reranking
            reranked = self.reranker.rerank(query, retrieved)[:5]

            # ✅ Context building
            context = "\n\n".join([r["text"] for r in reranked])

            prompt = f"""
Use ONLY the provided context.
If answer not found, say: "I don't have enough information."

Context:
{context}

Question:
{query}
"""

            # ✅ LLM call
            answer = self.llm.generate(prompt).strip()

            # ✅ Faithfulness
            try:
                faith = round(
                    faithfulness_score(query, context, answer), 3
                )
            except Exception as e:
                logger.error(f"Faithfulness error: {e}")
                faith = 0.0

            # 🔥 Track faithfulness
            FAITHFULNESS_SCORE.observe(faith)

            latency = round(time.time() - start, 3)

            result = {
                "answer": answer,
                "sources": list({r["source"] for r in reranked}),
                "latency_seconds": latency,
                "faithfulness": faith,
                "cached": False
            }

            # ✅ Cache store
            self.cache[query] = result.copy()

            logger.info(f"Latency: {latency}s | Faithfulness: {faith}")

            return result

        except Exception as e:
            logger.exception(f"RAG pipeline error: {e}")

            return {
                "answer": "⚠️ Internal error occurred.",
                "sources": [],
                "latency_seconds": 0,
                "faithfulness": 0.0,
                "cached": False
            }

    # ==========================================================
    # STREAMING RESPONSE
    # ==========================================================
    def stream_answer(self, query: str) -> Generator[str, None, None]:

        query = query.strip()
        logger.info(f"Streaming query: {query}")

        try:
            retrieved = self.retriever.retrieve(query, top_k=5)

            if not retrieved:
                yield "No relevant information found."
                return

            reranked = self.reranker.rerank(query, retrieved)[:5]

            context = "\n\n".join([r["text"] for r in reranked])

            prompt = f"""
Use ONLY the provided context.
If answer not found, say: "I don't have enough information."

Context:
{context}

Question:
{query}
"""

            # ✅ Stream tokens safely
            for token in self.llm.stream_generate(prompt):
                yield token

        except Exception as e:
            logger.exception(f"Streaming error: {e}")
            yield "⚠️ Streaming failed."


# ==========================================================
# ✅ SINGLETON
# ==========================================================
rag_service = RAGService()


# ==========================================================
# ✅ COMPATIBILITY WRAPPER
# ==========================================================
def answer_query(query: str):
    return rag_service.answer_query(query)

from prometheus_client import Counter, Histogram

# ✅ Cache hits
CACHE_HITS = Counter(
    "rag_cache_hits_total",
    "Number of cache hits"
)

# ✅ Faithfulness distribution
FAITHFULNESS_SCORE = Histogram(
    "rag_faithfulness_score",
    "Faithfulness score distribution"
)

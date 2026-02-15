import numpy as np
from ingestion.document_loader import load_documents
from core.chunking.text_chunker import chunk_text
from core.embeddings.embedding_model import EmbeddingModel
from vectorstore.faiss_store import FAISSStore
from elasticsearch import Elasticsearch, helpers

# =============================
# Configuration
# =============================
DATA_PATH = "data/raw_docs"
ES_HOST = "http://localhost:9200"
ES_INDEX = "rag_chunks"

# =============================
# Load & Chunk Documents
# =============================
documents = load_documents(DATA_PATH)
embedder = EmbeddingModel()

all_chunks = []
all_metadata = []

print("Loading and chunking documents...")

for doc in documents:
    chunks = chunk_text(doc["text"])

    for chunk in chunks:
        all_chunks.append(chunk)
        all_metadata.append({
            "text": chunk,
            "source": doc["source"]
        })

print(f"Total chunks created: {len(all_chunks)}")

# =============================
# Create Embeddings
# =============================
print("Generating embeddings...")
embeddings = embedder.embed(all_chunks)

# =============================
# FAISS Indexing
# =============================
print("Building FAISS index...")
store = FAISSStore(dim=embeddings.shape[1])
store.add(np.array(embeddings), all_metadata)
store.save()

print("✅ FAISS index built successfully")

# =============================
# Elasticsearch Indexing
# =============================
print("Connecting to Elasticsearch...")

es = Elasticsearch(
    hosts=[ES_HOST],
    verify_certs=False,
    request_timeout=30
)

try:
    info = es.info()
    print("Connected to Elasticsearch:", info["version"]["number"])
except Exception as e:
    raise Exception(f"❌ Cannot connect to Elasticsearch: {e}")

# Create index if not exists
if not es.indices.exists(index=ES_INDEX):
    es.indices.create(
        index=ES_INDEX,
        mappings={
            "properties": {
                "text": {"type": "text"},
                "source": {"type": "keyword"}
            }
        }
    )
    print("Elasticsearch index created.")

# Bulk insert
actions = [
    {
        "_index": ES_INDEX,
        "_source": meta
    }
    for meta in all_metadata
]

helpers.bulk(es, actions)

print("✅ Elasticsearch indexing complete.")
print("🚀 Hybrid RAG index ready.")

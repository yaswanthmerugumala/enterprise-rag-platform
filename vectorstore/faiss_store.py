import faiss
import json
import numpy as np
from app.config import FAISS_INDEX_PATH, METADATA_PATH

class FAISSStore:

    def __init__(self, dim):
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, metadatas):
        self.index.add(embeddings)
        self.metadata.extend(metadatas)

    def save(self):
        faiss.write_index(self.index, FAISS_INDEX_PATH)
        with open(METADATA_PATH, "w") as f:
            json.dump(self.metadata, f)

    @staticmethod
    def load():
        index = faiss.read_index(FAISS_INDEX_PATH)
        with open(METADATA_PATH, "r") as f:
            metadata = json.load(f)

        return index, metadata

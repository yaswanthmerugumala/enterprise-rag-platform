from elasticsearch import Elasticsearch

class SparseRetriever:

    def __init__(self):
        # self.es = Elasticsearch("http://elasticsearch:9200")
        self.es = Elasticsearch("http://localhost:9200")
        self.index_name = "rag_chunks"

    def retrieve(self, query, top_k=5):

        body = {
            "query": {
                "match": {
                    "text": query
                }
            }
        }

        response = self.es.search(
            index=self.index_name,
            body=body,
            size=top_k
        )

        results = []
        for hit in response["hits"]["hits"]:
            results.append({
                "text": hit["_source"]["text"],
                "source": hit["_source"]["source"],
                "score": hit["_score"]
            })

        return results

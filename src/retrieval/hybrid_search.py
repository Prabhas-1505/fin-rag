import asyncio
import os
from qdrant_client import AsyncQdrantClient
from sentence_transformers import SentenceTransformer
import pickle
from typing import List, Dict, Any, Optional

class HybridSearcher:
    def __init__(self,
                 bm25_index_path: str = "bm25_index.pkl",
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 qdrant_path: Optional[str] = None,
                 collection_name: str = "documents",
                 embedding_model_name: str = "BAAI/bge-large-en-v1.5"):

        self.collection_name = collection_name
        # QDRANT_LOCAL_PATH enables Qdrant's embedded local-file mode, so this
        # runs without a Docker/Qdrant server. Falls back to host/port otherwise.
        qdrant_path = qdrant_path or os.environ.get("QDRANT_LOCAL_PATH")
        if qdrant_path:
            self.qdrant_client = AsyncQdrantClient(path=qdrant_path)
        else:
            self.qdrant_client = AsyncQdrantClient(host=qdrant_host, port=qdrant_port)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        
        try:
            with open(bm25_index_path, 'rb') as f:
                data = pickle.load(f)
                self.bm25 = data["bm25"]
                self.chunks = data["chunks"]
        except FileNotFoundError:
            print(f"Warning: BM25 index not found at {bm25_index_path}. Sparse search will fail.")
            self.bm25 = None
            self.chunks = []

    async def search_dense(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Searches Qdrant using dense embeddings.
        """
        query_vector = self.embedding_model.encode(query).tolist()
        
        response = await self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=top_k
        )
        
        # Format results: [{"chunk_id": id, "score": score, "payload": payload}, ...]
        return [
            {
                "chunk_id": res.id,
                "score": res.score,
                "payload": res.payload
            }
            for res in response.points
        ]

    async def search_sparse(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Searches the local BM25 index.
        """
        if not self.bm25:
            return []
            
        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top_k indices sorted by score descending
        top_indices = scores.argsort()[::-1][:top_k]
        
        return [
            {
                "chunk_id": int(idx),
                "score": float(scores[idx]),
                "payload": {**self.chunks[idx]["metadata"], "text": self.chunks[idx]["text"]}
            }
            for idx in top_indices if scores[idx] > 0
        ]

    def reciprocal_rank_fusion(self, dense_results: List[Dict], sparse_results: List[Dict], k: int = 60) -> List[Dict]:
        """
        Fuses rankings from dense and sparse search using Reciprocal Rank Fusion.
        RRF Score = 1 / (k + rank)
        """
        fused_scores = {}
        
        # Process dense results
        for rank, res in enumerate(dense_results):
            chunk_id = res["chunk_id"]
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {"score": 0.0, "payload": res["payload"]}
            fused_scores[chunk_id]["score"] += 1.0 / (k + rank + 1)
            
        # Process sparse results
        for rank, res in enumerate(sparse_results):
            chunk_id = res["chunk_id"]
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {"score": 0.0, "payload": res["payload"]}
            fused_scores[chunk_id]["score"] += 1.0 / (k + rank + 1)
            
        # Sort by fused score descending
        sorted_results = sorted(
            [{"chunk_id": cid, **data} for cid, data in fused_scores.items()],
            key=lambda x: x["score"],
            reverse=True
        )
        
        return sorted_results

    async def hybrid_search(self, query: str, top_k: int = 25) -> List[Dict]:
        """
        Executes concurrent search and applies RRF.
        """
        dense_task = asyncio.create_task(self.search_dense(query, top_k=top_k))
        sparse_task = asyncio.create_task(self.search_sparse(query, top_k=top_k))
        
        dense_results, sparse_results = await asyncio.gather(dense_task, sparse_task)
        
        fused_results = self.reciprocal_rank_fusion(dense_results, sparse_results)
        
        # Return top 20 or 25 chunks
        return fused_results[:top_k]

if __name__ == "__main__":
    print("HybridSearcher initialized.")

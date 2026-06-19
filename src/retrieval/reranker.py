from sentence_transformers import CrossEncoder
from typing import List, Dict, Any

class Reranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-large"):
        """
        Initializes the cross-encoder for re-ranking retrieved chunks.
        """
        # We might use bge-reranker-base for faster local testing
        self.cross_encoder = CrossEncoder(model_name)

    def rerank(self, query: str, retrieved_chunks: List[Dict[str, Any]], top_k: int = 4) -> List[Dict[str, Any]]:
        """
        Re-ranks a list of candidate chunks against the user query.
        Discards everything except the top_k (e.g., top 4) chunks.
        """
        if not retrieved_chunks:
            return []

        # Assuming retrieved_chunks have the actual text in payload or we have the chunks list to lookup
        # In our HybridSearcher, the payload contains the chunk metadata, but we also need the text.
        # Let's assume the payload includes the original 'text' field (which we should add during ingestion).
        
        pairs = []
        for chunk in retrieved_chunks:
            # We expect chunk["payload"]["text"] to contain the raw chunk text
            text = chunk.get("payload", {}).get("text", "")
            pairs.append([query, text])
            
        # Predict scores for the query-document pairs
        scores = self.cross_encoder.predict(pairs)
        
        # Add scores to the chunks
        for idx, chunk in enumerate(retrieved_chunks):
            chunk["rerank_score"] = float(scores[idx])
            
        # Sort by rerank score descending
        sorted_chunks = sorted(retrieved_chunks, key=lambda x: x["rerank_score"], reverse=True)
        
        # Return only top 4
        return sorted_chunks[:top_k]

if __name__ == "__main__":
    print("Reranker initialized.")

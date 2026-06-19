import redis
import json
from sentence_transformers import SentenceTransformer
from typing import Optional, Dict, Any, List, cast
import numpy as np

class SemanticCache:
    def __init__(self, 
                 redis_host: str = "localhost", 
                 redis_port: int = 6379,
                 embedding_model_name: str = "BAAI/bge-large-en-v1.5",
                 similarity_threshold: float = 0.95):
        """
        Initializes the semantic cache layer using Redis.
        Stores queries and their embeddings to bypass the agent loop for repeated questions.
        """
        self.redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=False)
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None
        self.similarity_threshold = similarity_threshold
        self._local_cache = {}

    @property
    def embedding_model(self):
        if self._embedding_model is None:
            print("[System] Initializing SentenceTransformer (Downloading models if first time)...")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model

    def get_cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculates the cosine similarity between two vectors.
        """
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def check_cache(self, query: str) -> Optional[str]:
        """
        Checks if a highly similar query exists in the cache.
        Returns the cached answer if similarity > threshold, otherwise None.
        """
        try:
            query_embedding = np.asarray(self.embedding_model.encode(query))
            
            # Check if Redis is online
            use_redis = True
            keys = []
            try:
                self.redis_client.ping()
                keys = cast(List[Any], self.redis_client.keys("cache:*"))
            except Exception as redis_err:
                print(f"Redis offline, checking local RAM fallback cache: {redis_err}")
                use_redis = False
            
            best_match = None
            highest_similarity = -1.0
            
            if use_redis:
                for key in keys:
                    data_bytes = cast(bytes, self.redis_client.get(key))
                    if not data_bytes:
                        continue
                        
                    parsed_data = json.loads(data_bytes)
                    cached_embedding = np.array(parsed_data["embedding"])
                    
                    similarity = self.get_cosine_similarity(query_embedding, cached_embedding)
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = parsed_data["answer"]
            else:
                # Read from local in-memory dict
                for key, parsed_data in self._local_cache.items():
                    cached_embedding = np.array(parsed_data["embedding"])
                    similarity = self.get_cosine_similarity(query_embedding, cached_embedding)
                    
                    if similarity > highest_similarity:
                        highest_similarity = similarity
                        best_match = parsed_data["answer"]
                        
            if highest_similarity >= self.similarity_threshold:
                print(f"[Semantic Cache HIT] Similarity: {highest_similarity:.4f}")
                return best_match
                
            print(f"[Semantic Cache MISS] Highest Similarity: {highest_similarity:.4f}")
            return None
            
        except Exception as e:
            print(f"Semantic Cache Check Error: {e}")
            return None

    def set_cache(self, query: str, answer: str):
        """
        Saves a query, its embedding, and the answer to Redis and local fallback.
        """
        try:
            query_embedding = self.embedding_model.encode(query).tolist()
            
            # Simple hash or counter for key
            key_id = hash(query) % 1000000
            key = f"cache:{key_id}"
            
            payload = {
                "query": query,
                "embedding": query_embedding,
                "answer": answer
            }
            
            # Always save to local RAM cache
            self._local_cache[key] = payload
            print(f"[Local Cache SET] Saved query response to local RAM cache.")
            
            # Attempt to save to Redis
            try:
                self.redis_client.setex(key, 86400, json.dumps(payload))
                print(f"[Redis Cache SET] Cached response for query: '{query}'")
            except Exception as redis_err:
                print(f"Redis offline, could not write to Redis storage: {redis_err}")
            
        except Exception as e:
            print(f"Semantic Cache Set Error: {e}")

if __name__ == "__main__":
    print("SemanticCache initialized.")

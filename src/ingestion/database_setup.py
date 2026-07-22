import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sentence_transformers import SentenceTransformer
import psycopg2
from rank_bm25 import BM25Okapi
import pickle
from typing import List, Dict, Any, Optional

class DatabaseManager:
    def __init__(self,
                 qdrant_host: str = "localhost",
                 qdrant_port: int = 6333,
                 qdrant_path: Optional[str] = None,
                 pg_dsn: str = os.environ.get("PG_DSN", "postgresql://admin:password@localhost:5432/rag_db"),
                 embedding_model_name: str = "BAAI/bge-large-en-v1.5"):

        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.pg_dsn = pg_dsn

        # Dense Vector Database (Qdrant) Client
        # QDRANT_LOCAL_PATH enables embedded local-file mode (no Docker/server needed)
        qdrant_path = qdrant_path or os.environ.get("QDRANT_LOCAL_PATH")
        if qdrant_path:
            self.qdrant_client = QdrantClient(path=qdrant_path)
        else:
            self.qdrant_client = QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
        
        self.embedding_model_name = embedding_model_name
        self._embedding_model = None
        # Default bge-large-en vector size to avoid loading model just to get the dimension
        self.vector_size = 1024 
        
    @property
    def embedding_model(self):
        if self._embedding_model is None:
            print("[System] Initializing SentenceTransformer (Downloading models if first time)...")
            self._embedding_model = SentenceTransformer(self.embedding_model_name)
        return self._embedding_model
        
    def setup_qdrant_collection(self, collection_name: str = "documents"):
        """
        Creates a Qdrant collection if it doesn't exist.
        """
        collections = [c.name for c in self.qdrant_client.get_collections().collections]
        if collection_name not in collections:
            self.qdrant_client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(
                    size=self.vector_size, 
                    distance=models.Distance.COSINE
                ),
            )
            print(f"Created Qdrant collection: {collection_name}")
        else:
            print(f"Qdrant collection {collection_name} already exists.")

    def setup_postgres_tables(self):
        """
        Connects to PostgreSQL and creates tabular data structures.
        """
        try:
            conn = psycopg2.connect(self.pg_dsn)
            cursor = conn.cursor()
            
            # Example schemas for SQL Analytics Specialist
            create_tables_sql = """
            CREATE TABLE IF NOT EXISTS companies (
                ticker VARCHAR(10) PRIMARY KEY,
                company_name VARCHAR(100) NOT NULL,
                sector VARCHAR(50),
                industry VARCHAR(50)
            );

            CREATE TABLE IF NOT EXISTS financial_metrics (
                id SERIAL PRIMARY KEY,
                ticker VARCHAR(10) REFERENCES companies(ticker),
                quarter VARCHAR(10),
                revenue_millions DECIMAL,
                net_income_millions DECIMAL,
                eps DECIMAL,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            """
            cursor.execute(create_tables_sql)
            conn.commit()
            cursor.close()
            conn.close()
            print("PostgreSQL tables created successfully.")
        except Exception as e:
            print(f"Error setting up PostgreSQL: {e}")

    def build_bm25_index(self, chunks: List[Dict[str, Any]], index_path: str = "bm25_index.pkl"):
        """
        Builds a sparse BM25 index over exact text chunks and saves it locally.
        """
        corpus = [chunk["text"] for chunk in chunks]
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        
        bm25 = BM25Okapi(tokenized_corpus)
        
        # Save the index and the mapping
        with open(index_path, 'wb') as f:
            pickle.dump({"bm25": bm25, "chunks": chunks}, f)
            
        print(f"BM25 index built with {len(chunks)} chunks and saved to {index_path}")

    def ingest_chunks_to_qdrant(self, chunks: List[Dict[str, Any]], collection_name: str = "documents"):
        """
        Embeds text chunks and uploads them to Qdrant.
        """
        if not chunks:
            print("No chunks to ingest.")
            return

        texts = [chunk["text"] for chunk in chunks]
        metadatas = [chunk["metadata"] for chunk in chunks]

        print(f"Generating embeddings for {len(texts)} chunks...")
        embeddings = self.embedding_model.encode(texts).tolist()

        # Prepare Qdrant points. Payload must carry the raw text too (not just
        # metadata) since the reranker reads payload["text"] downstream.
        points = [
            models.PointStruct(
                id=idx,
                vector=embedding,
                payload={**meta, "text": text}
            )
            for idx, (embedding, meta, text) in enumerate(zip(embeddings, metadatas, texts))
        ]
        
        print("Uploading to Qdrant...")
        self.qdrant_client.upsert(
            collection_name=collection_name,
            points=points
        )
        print("Upload complete.")

if __name__ == "__main__":
    print("DatabaseManager initialized.")

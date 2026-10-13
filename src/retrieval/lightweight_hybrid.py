import pickle
from typing import Any, Dict, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity


class HashingTfidfEmbedder:
    """
    Memory-bounded text embedding via the hashing trick: HashingVectorizer
    stores no vocabulary (fixed-size feature space regardless of corpus size),
    unlike TfidfVectorizer which keeps a term->index dict that grows with the
    corpus. Stateless and picklable, so it doubles as a drop-in embedder for
    the semantic cache (no sentence-transformers/torch needed).
    """

    def __init__(self, n_features: int = 2**18):
        self.n_features = n_features
        self._vectorizer = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm=None
        )

    def encode(self, text: str) -> np.ndarray:
        vec = self._vectorizer.transform([text]).toarray()[0]
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def transform(self, texts: List[str]):
        return self._vectorizer.transform(texts)


class LightweightHybridSearcher:
    """
    Memory-light alternative to HybridSearcher: hashed-TFIDF cosine similarity
    (sparse, no stored vocabulary) fused with BM25 via reciprocal rank fusion,
    with no dense embedding model or cross-encoder reranker involved.
    """

    def __init__(self, index_path: str = "lightweight_index.pkl", n_features: int = 2**18):
        self.embedder = HashingTfidfEmbedder(n_features=n_features)
        self.tfidf_transformer: Optional[TfidfTransformer] = None
        self.doc_vectors = None
        self.bm25: Optional[BM25Okapi] = None
        self.chunks: List[Dict[str, Any]] = []

        try:
            with open(index_path, "rb") as f:
                data = pickle.load(f)
                self.chunks = data["chunks"]
                self.bm25 = data["bm25"]
                self.tfidf_transformer = data["tfidf_transformer"]
                self.doc_vectors = data["doc_vectors"]
        except FileNotFoundError:
            print(f"Warning: Lightweight index not found at {index_path}. Search will fail.")

    @classmethod
    def build_index(
        cls, chunks: List[Dict[str, Any]], index_path: str = "lightweight_index.pkl", n_features: int = 2**18
    ) -> "LightweightHybridSearcher":
        """
        Fits a corpus-wide IDF over the hashed term-frequency matrix and a BM25
        index over the same chunks, then pickles both for fast reload.
        """
        texts = [chunk["text"] for chunk in chunks]

        embedder = HashingTfidfEmbedder(n_features=n_features)
        hashed_tf = embedder.transform(texts)
        tfidf_transformer = TfidfTransformer()
        doc_vectors = tfidf_transformer.fit_transform(hashed_tf)

        tokenized_corpus = [doc.split(" ") for doc in texts]
        bm25 = BM25Okapi(tokenized_corpus)

        with open(index_path, "wb") as f:
            pickle.dump(
                {
                    "chunks": chunks,
                    "bm25": bm25,
                    "tfidf_transformer": tfidf_transformer,
                    "doc_vectors": doc_vectors,
                    "n_features": n_features,
                },
                f,
            )

        print(f"Lightweight hybrid index built with {len(chunks)} chunks and saved to {index_path}")

        searcher = cls.__new__(cls)
        searcher.embedder = embedder
        searcher.tfidf_transformer = tfidf_transformer
        searcher.doc_vectors = doc_vectors
        searcher.bm25 = bm25
        searcher.chunks = chunks
        return searcher

    def search_hashed_tfidf(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if self.doc_vectors is None or self.tfidf_transformer is None:
            return []

        hashed_query = self.embedder.transform([query])
        query_vector = self.tfidf_transformer.transform(hashed_query)
        scores = cosine_similarity(query_vector, self.doc_vectors)[0]

        top_indices = scores.argsort()[::-1][:top_k]
        return [
            {
                "chunk_id": int(idx),
                "score": float(scores[idx]),
                "payload": {**self.chunks[idx]["metadata"], "text": self.chunks[idx]["text"]},
            }
            for idx in top_indices
            if scores[idx] > 0
        ]

    def search_sparse(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        if not self.bm25:
            return []

        tokenized_query = query.split(" ")
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = scores.argsort()[::-1][:top_k]

        return [
            {
                "chunk_id": int(idx),
                "score": float(scores[idx]),
                "payload": {**self.chunks[idx]["metadata"], "text": self.chunks[idx]["text"]},
            }
            for idx in top_indices
            if scores[idx] > 0
        ]

    def reciprocal_rank_fusion(
        self, hashed_results: List[Dict], sparse_results: List[Dict], k: int = 60
    ) -> List[Dict]:
        fused_scores: Dict[int, Dict[str, Any]] = {}

        for rank, res in enumerate(hashed_results):
            chunk_id = res["chunk_id"]
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {"score": 0.0, "payload": res["payload"]}
            fused_scores[chunk_id]["score"] += 1.0 / (k + rank + 1)

        for rank, res in enumerate(sparse_results):
            chunk_id = res["chunk_id"]
            if chunk_id not in fused_scores:
                fused_scores[chunk_id] = {"score": 0.0, "payload": res["payload"]}
            fused_scores[chunk_id]["score"] += 1.0 / (k + rank + 1)

        return sorted(
            [{"chunk_id": cid, **data} for cid, data in fused_scores.items()],
            key=lambda x: x["score"],
            reverse=True,
        )

    def hybrid_search(self, query: str, top_k: int = 25) -> List[Dict]:
        hashed_results = self.search_hashed_tfidf(query, top_k=top_k)
        sparse_results = self.search_sparse(query, top_k=top_k)
        fused_results = self.reciprocal_rank_fusion(hashed_results, sparse_results)
        return fused_results[:top_k]


if __name__ == "__main__":
    print("LightweightHybridSearcher module loaded.")

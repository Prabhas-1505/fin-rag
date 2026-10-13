"""
Standalone subprocess target: builds ONE retrieval index over a cached chunk
set and runs a cached query set, then reports this process's own peak RSS.
Run in a fresh process (not imported) so each mode's memory measurement
isn't contaminated by the other mode or by the orchestrating script.
"""
import argparse
import json
import pickle
import platform
import resource
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["tfidf", "tfidf_densified", "hashing"], required=True)
    parser.add_argument("--chunks", default="data/eval_corpus/_chunks_cache.pkl")
    parser.add_argument("--queries", default="data/eval_corpus/_queries_cache.pkl")
    parser.add_argument("--n-features", type=int, default=2**18)
    parser.add_argument(
        "--synthetic-multiplier", type=int, default=1,
        help="Replicate the corpus N times, appending unique per-replica tokens to each "
             "chunk so vocabulary actually grows (not literal duplicates) - simulates a "
             "larger real-world corpus to find where vocabulary-based storage stops scaling.",
    )
    args = parser.parse_args()

    with open(args.chunks, "rb") as f:
        chunks = pickle.load(f)
    with open(args.queries, "rb") as f:
        queries = pickle.load(f)

    texts = [c["text"] for c in chunks]
    if args.synthetic_multiplier > 1:
        expanded = []
        for i in range(args.synthetic_multiplier):
            suffix = " ".join(f"syntoken{i}_{j}" for j in range(20))
            expanded.extend(f"{t} {suffix}" for t in texts)
        texts = expanded

    t0 = time.time()
    if args.mode in ("tfidf", "tfidf_densified"):
        # The naive baseline: sklearn's TfidfVectorizer keeps a vocabulary
        # dict (token -> column index) that grows with the corpus.
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity

        vectorizer = TfidfVectorizer()
        doc_vectors = vectorizer.fit_transform(texts)
        if args.mode == "tfidf_densified":
            # The actual common real-world mistake: calling .toarray() on the
            # sparse matrix (e.g. to feed a non-sparse-aware downstream step)
            # materializes every zero entry as a real float - this, not the
            # vectorizer choice itself, is what tends to cause OOM crashes.
            doc_vectors = doc_vectors.toarray()
        build_seconds = time.time() - t0

        t1 = time.time()
        for q in queries:
            qv = vectorizer.transform([q])
            if args.mode == "tfidf_densified":
                qv = qv.toarray()
            _ = cosine_similarity(qv, doc_vectors)
        query_seconds = time.time() - t1
    else:
        # The fix: HashingVectorizer has a fixed feature space, no stored
        # vocabulary regardless of corpus size.
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.metrics.pairwise import cosine_similarity
        from src.retrieval.lightweight_hybrid import HashingTfidfEmbedder

        embedder = HashingTfidfEmbedder(n_features=args.n_features)
        hashed = embedder.transform(texts)
        transformer = TfidfTransformer()
        doc_vectors = transformer.fit_transform(hashed)
        build_seconds = time.time() - t0

        t1 = time.time()
        for q in queries:
            hq = embedder.transform([q])
            qv = transformer.transform(hq)
            _ = cosine_similarity(qv, doc_vectors)
        query_seconds = time.time() - t1

    peak_rss_raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS reports ru_maxrss in bytes, Linux in KB.
    peak_rss_mb = peak_rss_raw / (1024 * 1024) if platform.system() == "Darwin" else peak_rss_raw / 1024

    print(json.dumps({
        "mode": args.mode,
        "synthetic_multiplier": args.synthetic_multiplier,
        "peak_rss_mb": round(peak_rss_mb, 1),
        "build_seconds": round(build_seconds, 3),
        "query_seconds_total": round(query_seconds, 3),
        "num_chunks": len(texts),
        "num_queries": len(queries),
    }))


if __name__ == "__main__":
    main()

"""
CLI entrypoint to ingest a PDF into the RAG system: parses it, chunks it,
builds the local BM25 sparse index, and upserts embeddings into Qdrant.

Usage:
    PYTHONPATH=. python src/ingestion/run_ingestion.py path/to/file.pdf --source-id my_doc
"""
import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.document_processor import DocumentProcessor
from src.ingestion.database_setup import DatabaseManager


def main():
    parser = argparse.ArgumentParser(description="Ingest a PDF into the hybrid search index.")
    parser.add_argument("pdf_path", help="Path to the PDF file to ingest.")
    parser.add_argument("--source-id", default=None, help="Identifier for this document (defaults to filename).")
    parser.add_argument("--collection", default="documents", help="Qdrant collection name.")
    parser.add_argument("--bm25-path", default="bm25_index.pkl", help="Where to save the BM25 index.")
    args = parser.parse_args()

    source_id = args.source_id or args.pdf_path.split("/")[-1]

    print(f"[1/4] Parsing and chunking {args.pdf_path} ...")
    processor = DocumentProcessor()
    chunks = processor.process_pdf_to_chunks(args.pdf_path, source_id=source_id)
    print(f"      -> {len(chunks)} chunks produced.")

    if not chunks:
        print("No chunks produced, aborting.")
        sys.exit(1)

    manager = DatabaseManager()

    print(f"[2/4] Building BM25 sparse index -> {args.bm25_path}")
    manager.build_bm25_index(chunks, index_path=args.bm25_path)

    print(f"[3/4] Ensuring Qdrant collection '{args.collection}' exists ...")
    manager.setup_qdrant_collection(collection_name=args.collection)

    print(f"[4/4] Embedding and upserting {len(chunks)} chunks into Qdrant ...")
    manager.ingest_chunks_to_qdrant(chunks, collection_name=args.collection)

    print("\nDone. Real document is now searchable via the hybrid search pipeline.")


if __name__ == "__main__":
    main()

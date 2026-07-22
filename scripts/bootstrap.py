"""
One-shot startup bootstrap for a hosted deployment (e.g. Render): creates the
Postgres schema if missing, seeds it with real financial data, and ingests
the sample document into the embedded Qdrant index. Safe to run on every
boot - it truncates and re-seeds rather than erroring on existing data.
"""
import warnings

warnings.simplefilter("ignore")
from dotenv import load_dotenv

load_dotenv()

from src.ingestion.database_setup import DatabaseManager
from src.ingestion.document_processor import DocumentProcessor
import src.ingestion.seed_real_financials as seed_real_financials


def main():
    print("[bootstrap] Setting up Postgres schema...")
    manager = DatabaseManager()
    manager.setup_postgres_tables()

    print("[bootstrap] Seeding real financial data...")
    seed_real_financials.main()

    print("[bootstrap] Ensuring Qdrant collection + sample document ingested...")
    manager.setup_qdrant_collection(collection_name="documents")
    processor = DocumentProcessor()
    chunks = processor.process_pdf_to_chunks(
        "data/documents/illegal_logging.pdf", source_id="illegal_logging_paper"
    )
    manager.build_bm25_index(chunks, index_path="bm25_index.pkl")
    manager.ingest_chunks_to_qdrant(chunks, collection_name="documents")

    print("[bootstrap] Done.")


if __name__ == "__main__":
    main()

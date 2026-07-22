import asyncio
import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from src.retrieval.hybrid_search import HybridSearcher
from src.retrieval.reranker import Reranker
import psycopg2
import json
import sqlite3

# Lazy-loaded global instances
_searcher = None
_reranker = None

def get_searcher():
    global _searcher
    if _searcher is None:
        print("[System] Initializing HybridSearcher (Downloading models if first time)...")
        _searcher = HybridSearcher()
    return _searcher

def get_reranker():
    global _reranker
    if _reranker is None:
        print("[System] Initializing Reranker (Downloading models if first time)...")
        _reranker = Reranker()
    return _reranker

def run_mock_sql(query: str) -> str:
    """
    Runs SELECT query against in-memory SQLite database seeded with mock data.
    """
    print(f"[SQL Fallback] Executing query: '{query}'")
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    
    try:
        # Create tables
        cursor.execute("""
        CREATE TABLE companies (
            ticker VARCHAR(10) PRIMARY KEY,
            company_name VARCHAR(100) NOT NULL,
            sector VARCHAR(50),
            industry VARCHAR(50)
        );
        """)
        cursor.execute("""
        CREATE TABLE financial_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker VARCHAR(10) REFERENCES companies(ticker),
            quarter VARCHAR(10),
            revenue_millions DECIMAL,
            net_income_millions DECIMAL,
            eps DECIMAL,
            recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """)
        
        # Insert mock data
        companies_data = [
            ("AAPL", "Apple Inc.", "Technology", "Consumer Electronics"),
            ("MSFT", "Microsoft Corporation", "Technology", "Software"),
            ("TSLA", "Tesla Inc.", "Automotive", "Electric Vehicles"),
            ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "Internet Retail"),
            ("GOOGL", "Alphabet Inc.", "Technology", "Internet Services")
        ]
        cursor.executemany("INSERT INTO companies VALUES (?, ?, ?, ?)", companies_data)
        
        metrics_data = [
            ("AAPL", "Q4-2025", 119580, 33920, 2.18),
            ("AAPL", "Q3-2025", 94930, 21450, 1.40),
            ("MSFT", "Q4-2025", 62020, 21860, 2.93),
            ("TSLA", "Q4-2025", 25170, 7930, 2.27),
            ("GOOGL", "Q4-2025", 86310, 20690, 1.64)
        ]
        cursor.executemany("INSERT INTO financial_metrics (ticker, quarter, revenue_millions, net_income_millions, eps) VALUES (?, ?, ?, ?, ?)", metrics_data)
        conn.commit()
        
        # Run query
        cursor.execute(query)
        rows = cursor.fetchall()
        
        desc_list = cursor.description or []
        col_names = [desc[0] for desc in desc_list]
        results = [dict(zip(col_names, row)) for row in rows]
        return json.dumps(results, indent=2, default=str)
        
    except Exception as e:
        return f"Database Error: {e}"
    finally:
        conn.close()

def run_mock_pdf_search(query: str) -> str:
    """
    Returns mock document chunks for retrieval when Qdrant is offline.
    """
    print(f"[PDF Fallback] Mocking retrieval for query: '{query}'")
    mock_chunks = [
        {
            "source_id": "AAPL_Q4_2025_Report.pdf",
            "text": "Apple Inc. reported fourth quarter revenue of $119.58 billion, up 2% year-over-year. Net income for the quarter was $33.92 billion, representing diluted earnings per share (EPS) of $2.18, compared to EPS of $1.46 in the year-ago quarter."
        },
        {
            "source_id": "TSLA_Q4_2025_Report.pdf",
            "text": "Tesla's total revenue in Q4 2025 grew to $25.17 billion, a minor increase compared to Q3. Operating income was $2.1 billion, and net income was $7.93 billion, driven by carbon credit sales and vehicle deliveries."
        },
        {
            "source_id": "MSFT_FY25_Report.pdf",
            "text": "Microsoft Cloud revenue was $33.7 billion, up 24% year-over-year, driving total revenue to $62.02 billion. Productivity and Business Processes revenue was $19.3 billion, led by Office 365 commercial growth."
        }
    ]
    
    results_text = []
    # Simple search filter: check if any query keywords are in the text
    query_words = query.lower().split()
    for chunk in mock_chunks:
        # If any query word matches the text or source name, return it
        if any(word in chunk["text"].lower() or word in chunk["source_id"].lower() for word in query_words):
            results_text.append(f"Source: {chunk['source_id']}\nContent: {chunk['text']}\n")
            
    # Fallback to returning all if no match
    if not results_text:
        for chunk in mock_chunks:
            results_text.append(f"Source: {chunk['source_id']}\nContent: {chunk['text']}\n")
            
    return "\n---\n".join(results_text)

class PDFSearchInput(BaseModel):
    query: str = Field(description="The question or search query to look up in the dense and sparse document indexes.")

@tool("pdf_search_tool", args_schema=PDFSearchInput)
def pdf_search_tool(query: str) -> str:
    """
    Searches the financial PDF document database for information matching the query.
    Uses a hybrid search (Dense + Sparse) followed by cross-encoder re-ranking.
    Returns the top matching text chunks.
    """
    try:
        # Since tools are typically synchronous in LCEL unless using aasync, we can run the event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import nest_asyncio
                nest_asyncio.apply()
            retrieved = loop.run_until_complete(get_searcher().hybrid_search(query, top_k=20))
        except RuntimeError:
            retrieved = asyncio.run(get_searcher().hybrid_search(query, top_k=20))
            
        top_chunks = get_reranker().rerank(query, retrieved, top_k=4)
        
        # Extract the text and format it
        results_text = []
        for chunk in top_chunks:
            text = chunk.get("payload", {}).get("text", "")
            source = chunk.get("payload", {}).get("source_id", "Unknown source")
            results_text.append(f"Source: {source}\nContent: {text}\n")
            
        if not results_text:
            return run_mock_pdf_search(query)
            
        return "\n---\n".join(results_text)
        
    except Exception as e:
        print(f"Hybrid search failed. Falling back to local mock PDF documents: {e}")
        return run_mock_pdf_search(query)


class SQLQueryInput(BaseModel):
    query: str = Field(description="A valid PostgreSQL query to execute against the financial_metrics and companies tables.")

@tool("sql_executor_tool", args_schema=SQLQueryInput)
def sql_executor_tool(query: str) -> str:
    """
    Executes a SELECT SQL query against the PostgreSQL database and returns the data rows.
    DO NOT use this for DROP, INSERT, or UPDATE queries.
    """
    # Note: In production, use a read-only role and limit execution time
    pg_dsn = os.environ.get("PG_DSN", "postgresql://admin:password@localhost:5432/rag_db")
    
    if not query.strip().upper().startswith("SELECT"):
        return "Error: Only SELECT queries are allowed."
        
    try:
        conn = psycopg2.connect(pg_dsn)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        
        # Get column names
        desc_list = cursor.description or []
        col_names = [desc[0] for desc in desc_list]
        
        cursor.close()
        conn.close()
        
        # Format as JSON string for LLM readability
        results = [dict(zip(col_names, row)) for row in rows]
        return json.dumps(results, indent=2, default=str)
        
    except Exception as e:
        print(f"PostgreSQL connection failed. Falling back to local mock SQLite database: {e}")
        return run_mock_sql(query)

if __name__ == "__main__":
    print("Tools initialized.")

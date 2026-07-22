"""
Seeds the Postgres companies/financial_metrics tables with real, sourced
quarterly figures (stockanalysis.com, fetched 2026-07-22) for 5 real
companies, replacing the placeholder mock data used when Postgres was
unreachable.

Usage: PYTHONPATH=. python src/ingestion/seed_real_financials.py
"""
import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

PG_DSN = os.environ.get("PG_DSN", "postgresql://admin:password@localhost:5432/rag_db")

COMPANIES = [
    ("AAPL", "Apple Inc.", "Technology", "Consumer Electronics"),
    ("MSFT", "Microsoft Corporation", "Technology", "Software"),
    ("TSLA", "Tesla Inc.", "Automotive", "Electric Vehicles"),
    ("AMZN", "Amazon.com Inc.", "Consumer Discretionary", "Internet Retail"),
    ("GOOGL", "Alphabet Inc.", "Technology", "Internet Services"),
]

# (ticker, quarter, revenue_millions, net_income_millions, eps)
# Source: stockanalysis.com quarterly financials, fetched 2026-07-22
METRICS = [
    ("AAPL", "Q2-2026", 111184, 29578, 2.01),
    ("AAPL", "Q1-2026", 143756, 42097, 2.84),
    ("AAPL", "Q4-2025", 102466, 27466, 1.85),
    ("AAPL", "Q3-2025", 94036, 23434, 1.57),

    ("MSFT", "Q3-2026", 82886, 31778, 4.27),
    ("MSFT", "Q2-2026", 81273, 38458, 5.16),
    ("MSFT", "Q1-2026", 77673, 27747, 3.72),
    ("MSFT", "Q4-2025", 76441, 27233, 3.65),

    ("TSLA", "Q1-2026", 22387, 477, 0.13),
    ("TSLA", "Q4-2025", 24901, 840, 0.24),
    ("TSLA", "Q3-2025", 28095, 1373, 0.39),
    ("TSLA", "Q2-2025", 22496, 1172, 0.33),

    ("AMZN", "Q1-2026", 181519, 30255, 2.78),
    ("AMZN", "Q4-2025", 213386, 21192, 1.95),
    ("AMZN", "Q3-2025", 180169, 21187, 1.95),
    ("AMZN", "Q2-2025", 167702, 18164, 1.68),

    ("GOOGL", "Q1-2026", 109896, 62578, 5.11),
    ("GOOGL", "Q4-2025", 113829, 34455, 2.82),
    ("GOOGL", "Q3-2025", 102346, 34979, 2.87),
    ("GOOGL", "Q2-2025", 96428, 28196, 2.31),
]


def main():
    conn = psycopg2.connect(PG_DSN)
    cur = conn.cursor()

    cur.execute("TRUNCATE TABLE financial_metrics, companies RESTART IDENTITY CASCADE;")

    cur.executemany(
        "INSERT INTO companies (ticker, company_name, sector, industry) VALUES (%s, %s, %s, %s)",
        COMPANIES,
    )
    cur.executemany(
        """INSERT INTO financial_metrics (ticker, quarter, revenue_millions, net_income_millions, eps)
           VALUES (%s, %s, %s, %s, %s)""",
        METRICS,
    )

    conn.commit()
    cur.execute("SELECT COUNT(*) FROM companies;")
    n_companies = cur.fetchone()[0]
    cur.execute("SELECT COUNT(*) FROM financial_metrics;")
    n_metrics = cur.fetchone()[0]
    cur.close()
    conn.close()

    print(f"Seeded {n_companies} companies and {n_metrics} real quarterly financial records.")


if __name__ == "__main__":
    main()

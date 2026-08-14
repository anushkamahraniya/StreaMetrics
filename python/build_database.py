"""
StreaMetrics - Database Builder

Executes the DDL in sql/01_schema.sql against data/streaming_analytics.db and
loads the generated CSVs into the users, videos, and watch_events tables.

This environment does not ship the `sqlite3` CLI binary, so the SQLite CLI
dot-commands (.mode, .import) in sql/01_schema.sql cannot be executed
directly. This script performs the equivalent work using Python's built-in
sqlite3 module: it re-executes the same CREATE TABLE / CREATE INDEX
statements from 01_schema.sql, then loads each CSV via csv.DictReader.

Where the `sqlite3` CLI is available, sql/01_schema.sql can be run directly:
    sqlite3 data/streaming_analytics.db < sql/01_schema.sql
"""

import csv
import os
import re
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
SQL_DIR = os.path.join(BASE_DIR, "sql")
DB_PATH = os.path.join(DATA_DIR, "streaming_analytics.db")
SCHEMA_PATH = os.path.join(SQL_DIR, "01_schema.sql")

TABLES = ["users", "videos", "watch_events"]


def extract_ddl_statements(schema_sql: str):
    """Pull only the CREATE TABLE / DROP TABLE / CREATE INDEX / PRAGMA
    statements out of the schema file, skipping sqlite3 CLI dot-commands
    (.mode, .import) and the trailing sanity-check SELECT."""
    # Drop any line starting with a dot-command
    lines = [ln for ln in schema_sql.splitlines() if not ln.strip().startswith(".")]
    cleaned = "\n".join(lines)

    # Strip the final sanity-check SELECT ... UNION ALL ... block
    cleaned = re.split(r"--\s*Sanity check", cleaned, flags=re.IGNORECASE)[0]

    statements = [s.strip() for s in cleaned.split(";") if s.strip()]
    return statements


def main():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
        schema_sql = f.read()

    ddl_statements = extract_ddl_statements(schema_sql)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for stmt in ddl_statements:
        cur.execute(stmt)

    # Loading CSV data
    for table in TABLES:
        csv_path = os.path.join(DATA_DIR, f"{table}.csv")
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            header = next(reader)
            placeholders = ", ".join(["?"] * len(header))
            columns = ", ".join(header)
            insert_sql = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"
            cur.executemany(insert_sql, reader)

    conn.commit()

    print("Database build complete:", DB_PATH)
    for table in TABLES:
        count = cur.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"  {table:<15} -> {count:,} rows")

    conn.close()


if __name__ == "__main__":
    main()

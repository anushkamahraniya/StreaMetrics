"""
StreaMetrics - Analytics Query Runner

Executes each query block in sql/02_analytics_queries.sql against
data/streaming_analytics.db and prints formatted results to the terminal.

This environment does not ship the `sqlite3` CLI binary (only Python's
built-in sqlite3 module), so this runner reproduces `sqlite3 ... < 02_analytics_queries.sql`
using Python. Where the CLI is available, the .sql file can be run directly:
    sqlite3 data/streaming_analytics.db < sql/02_analytics_queries.sql
"""

import os
import re
import sqlite3

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "streaming_analytics.db")
SQL_PATH = os.path.join(BASE_DIR, "sql", "02_analytics_queries.sql")


def load_query_blocks(sql_text: str):
    """Split the file into (title, statement) pairs using the '-- QUERY N: TITLE'
    banner comments as delimiters, skipping CLI dot-commands."""
    lines = [ln for ln in sql_text.splitlines() if not ln.strip().startswith(".")]
    cleaned = "\n".join(lines)

    blocks = re.split(r"-- QUERY \d+:\s*", cleaned)[1:]  # drop preamble before first query
    queries = []
    for block in blocks:
        title_line, _, rest = block.partition("\n")
        title = title_line.strip()
        # Strip remaining comment lines (---- style banners and descriptive comments)
        stmt_lines = [ln for ln in rest.splitlines() if not ln.strip().startswith("--")]
        statement = "\n".join(stmt_lines).strip().rstrip(";")
        queries.append((title, statement))
    return queries


def print_table(headers, rows, max_col_width=22):
    def fmt(val):
        s = "" if val is None else str(val)
        return s if len(s) <= max_col_width else s[: max_col_width - 1] + "…"

    str_rows = [[fmt(v) for v in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, val in enumerate(row):
            widths[i] = max(widths[i], len(val))

    def print_row(row):
        print("  ".join(val.ljust(widths[i]) for i, val in enumerate(row)))

    print_row(headers)
    print_row(["-" * w for w in widths])
    for row in str_rows:
        print_row(row)


def main():
    with open(SQL_PATH, "r", encoding="utf-8") as f:
        sql_text = f.read()

    queries = load_query_blocks(sql_text)

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    for title, statement in queries:
        print("=" * 100)
        print(title)
        print("=" * 100)
        cur.execute(statement)
        headers = [d[0] for d in cur.description]
        rows = cur.fetchall()
        print_table(headers, rows)
        print(f"\n({len(rows)} rows)\n")

    conn.close()


if __name__ == "__main__":
    main()

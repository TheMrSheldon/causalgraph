#!/usr/bin/env python
"""
Development utility: inspect the parquet file schema and sample rows.

Usage:
    python scripts/inspect_parquet.py
    python scripts/inspect_parquet.py --sample 20
    python scripts/inspect_parquet.py --causal-sample  # Preview regex results
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import duckdb


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect rscience-submissions.parquet")
    parser.add_argument("--parquet", default="rscience-submissions.parquet")
    parser.add_argument("--sample", type=int, default=10, help="Number of sample rows")
    parser.add_argument("--causal-sample", action="store_true",
                        help="Show sample titles through regex identifier")
    args = parser.parse_args()

    con = duckdb.connect()

    print(f"=== Schema: {args.parquet} ===")
    schema = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{args.parquet}')").fetchall()
    for col_name, col_type, *_ in schema:
        print(f"  {col_name:<40} {col_type}")

    total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{args.parquet}')").fetchone()[0]
    print(f"\n=== Total rows: {total:,} ===\n")

    print(f"=== Sample {args.sample} titles ===")
    rows = con.execute(
        f"""
        SELECT id, title, score, num_comments, created_utc
        FROM read_parquet('{args.parquet}')
        WHERE title IS NOT NULL AND score > 0
        ORDER BY score DESC
        LIMIT {args.sample}
        """
    ).fetchall()
    for row in rows:
        print(f"  [{row[2]:>6} pts] {row[1][:100]}")

    if args.causal_sample:
        print("\n=== Causal sample (regex identifier) ===")
        from pipeline.step1_identification.regex_identifier import RegexIdentifier
        from pipeline.parquet_reader import ParquetReader

        reader = ParquetReader(args.parquet, min_score=1)
        identifier = RegexIdentifier()
        batch = next(reader.iter_batches(1000))
        causal = identifier.identify(batch)
        print(f"  {len(causal)}/{len(batch)} causal in first batch ({len(causal)/len(batch)*100:.1f}%)")
        for p in causal[:20]:
            print(f"  [{p.score:>5} pts] {p.title[:100]}")

    con.close()


if __name__ == "__main__":
    main()

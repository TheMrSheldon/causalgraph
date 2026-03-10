"""
PyArrow-backed Parquet reader.

Uses pyarrow.parquet.ParquetFile.iter_batches() for true streaming —
reads only the needed columns and yields small row-group batches without
loading the entire 214MB file into memory at once.
"""
from __future__ import annotations

from typing import Generator

import pyarrow as pa
import pyarrow.parquet as pq

from .protocols import Post

_COLUMNS = ["id", "title", "score", "num_comments", "created_utc", "author", "url", "permalink"]


def _row_to_post(row: dict) -> Post | None:
    title = row.get("title")
    if not title or not isinstance(title, str) or len(title.strip()) <= 10:
        return None
    score = int(row.get("score") or 0)
    return Post(
        id=str(row["id"]),
        title=title,
        score=score,
        num_comments=int(row.get("num_comments") or 0),
        created_utc=int(row.get("created_utc") or 0),
        author=str(row["author"]) if row.get("author") else None,
        url=str(row["url"]) if row.get("url") else None,
        permalink=str(row["permalink"]) if row.get("permalink") else None,
    )


class ParquetReader:
    def __init__(self, parquet_path: str, min_score: int = 1) -> None:
        self.parquet_path = parquet_path
        self.min_score = min_score

    def count(self) -> int:
        """Total rows in the file (unfiltered, fast metadata read)."""
        return pq.read_metadata(self.parquet_path).num_rows

    def iter_batches(self, batch_size: int = 5000) -> Generator[list[Post], None, None]:
        """
        True streaming: reads one Parquet row group at a time via iter_batches().
        Memory footprint is proportional to one batch, not the whole file.
        """
        pf = pq.ParquetFile(self.parquet_path)
        buf: list[Post] = []

        for record_batch in pf.iter_batches(batch_size=batch_size, columns=_COLUMNS):
            table = pa.Table.from_batches([record_batch])
            for i in range(table.num_rows):
                row = {col: table.column(col)[i].as_py() for col in _COLUMNS}
                if (row.get("score") or 0) < self.min_score:
                    continue
                post = _row_to_post(row)
                if post:
                    buf.append(post)
                if len(buf) >= batch_size:
                    yield buf
                    buf = []

        if buf:
            yield buf

    def sample(self, n: int = 100) -> list[Post]:
        """Return the first n qualifying posts (for debugging)."""
        posts: list[Post] = []
        for batch in self.iter_batches(batch_size=1000):
            posts.extend(batch)
            if len(posts) >= n:
                break
        return posts[:n]

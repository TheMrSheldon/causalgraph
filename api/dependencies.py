"""Shared FastAPI dependencies."""
from __future__ import annotations

import os
from functools import lru_cache

from api.db import GraphDatabase


@lru_cache(maxsize=1)
def get_db() -> GraphDatabase:
    db_path = os.environ.get("GRAPH_DB_PATH") or "data/pipeline.db"
    return GraphDatabase(db_path)

"""Shared FastAPI dependencies."""
from __future__ import annotations

import os
from functools import lru_cache

import yaml

from api.db import GraphDatabase


@lru_cache(maxsize=1)
def get_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_db() -> GraphDatabase:
    # GRAPH_DB_PATH env var overrides config.yaml so containers can mount the
    # database at a fixed path (e.g. /causalgraph/graph.db) without touching
    # the config file.
    db_path = os.environ.get("GRAPH_DB_PATH") or get_config()["api"]["db_path"]
    return GraphDatabase(db_path)

"""Shared FastAPI dependencies."""
from __future__ import annotations

from functools import lru_cache

import yaml

from pipeline.db import Database


@lru_cache(maxsize=1)
def get_config() -> dict:
    with open("config.yaml") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def get_db() -> Database:
    config = get_config()
    db_path = config["api"]["db_path"]
    return Database(db_path)

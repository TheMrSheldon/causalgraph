"""FastAPI application factory."""
from __future__ import annotations

import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.routers import clusters, graph, posts


def create_app() -> FastAPI:
    app = FastAPI(
        title="r/science Causal Graph API",
        description="Explore causal relationships extracted from r/science submissions.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    app.include_router(graph.router)
    app.include_router(clusters.router)
    app.include_router(posts.router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/api/version")
    def version() -> dict:
        return {
            "version": os.environ.get("BUILD_VERSION", "dev"),
            "build_date": os.environ.get("BUILD_DATE", "unknown"),
        }

    return app


app = create_app()

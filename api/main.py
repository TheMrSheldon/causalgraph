"""FastAPI application factory."""
from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.dependencies import get_config
from api.routers import clusters, graph, posts


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    # Eager-load config and DB connection on startup
    get_config()
    yield


def create_app() -> FastAPI:
    config = get_config()
    cors_origins = config.get("api", {}).get("cors_origins", ["*"])

    app = FastAPI(
        title="r/science Causal Graph API",
        description="Explore causal relationships extracted from r/science submissions.",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
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

    return app


app = create_app()

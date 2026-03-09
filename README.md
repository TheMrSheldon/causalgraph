# r/science Causal Graph

Extract causal relationships from 867K r/science submission titles and explore them as an interactive hierarchical graph.

## Quick Start

```bash
# 1. Install Python deps
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# 2. Run the pipeline (writes data/pipeline.db)
python scripts/run_pipeline.py

# 3. Start the backend API  (terminal 1)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. Start the pipeline server  (terminal 2)
uvicorn pipeline.server:app --host 0.0.0.0 --port 8001

# 5. Start the frontend  (terminal 3)
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** — double-click nodes to expand, click edges to see source posts.

The **Text Analyzer** feature (type any sentence to extract causal relations live) requires the pipeline server (step 4).

## Documentation

- [Architecture](docs/architecture.md) — system design with Mermaid diagrams
- [Getting Started](docs/getting-started.md) — detailed setup and configuration
- [Extending](docs/extending.md) — how to add new pipeline implementations
- [Graph Format](docs/graphformat.md) — SQLite schema contract between pipeline and API

## Structure

```
pipeline/           Python pipeline (3 pluggable steps) + pipeline REST server
api/                FastAPI backend (reads pipeline.db; no pipeline imports)
frontend/           React + Cytoscape.js UI
config.yaml         Swap implementations without code changes
data/pipeline.db    SQLite output (created by pipeline)
docs/               Architecture, getting-started, extending, graph format
```

## Services

| Service | Command | Port | Purpose |
|---------|---------|------|---------|
| Backend API | `uvicorn api.main:app` | 8000 | Serves graph data from `pipeline.db` |
| Pipeline server | `uvicorn pipeline.server:app --port 8001` | 8001 | Per-step REST API (text analyzer) |
| Frontend | `npm run dev` | 5173 | Interactive graph UI |

## Swapping components

Edit `config.yaml`:

```yaml
pipeline:
  step1_identification:
    implementation: "llm_anthropic"   # was "regex"
  step3_hierarchy:
    implementation: "tfidf_ward"      # was "embedding_hdbscan"
```

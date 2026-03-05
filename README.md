# r/science Causal Graph

Extract causal relationships from 867K r/science submission titles and explore them as an interactive hierarchical graph.

## Quick Start

```bash
# 1. Install Python deps
pip install -e ".[dev]"
python -m spacy download en_core_web_sm

# 2. Run the pipeline
python scripts/run_pipeline.py

# 3. Start the API
uvicorn api.main:app --reload --port 8000

# 4. Start the frontend (separate terminal)
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** — double-click nodes to expand, click edges to see source posts.

## Documentation

- [Architecture](docs/architecture.md) — system design with Mermaid diagrams
- [Getting Started](docs/getting-started.md) — detailed setup and configuration
- [Extending](docs/extending.md) — how to add new pipeline implementations

## Structure

```
pipeline/           Python pipeline (3 pluggable steps)
api/                FastAPI backend
frontend/           React + Cytoscape.js UI
config.yaml         Swap implementations without code changes
data/pipeline.db    SQLite output (created by pipeline)
```

## Swapping components

Edit `config.yaml`:

```yaml
pipeline:
  step1_identification:
    implementation: "llm_anthropic"   # was "regex"
  step3_hierarchy:
    implementation: "tfidf_ward"      # was "embedding_hdbscan"
```

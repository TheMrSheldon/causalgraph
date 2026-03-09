# r/science Causal Graph

Extract causal relationships from 867K r/science submission titles and explore them as an interactive hierarchical graph.

## Quick Start

```bash
# 1. Install Python deps
pip install -r api/requirements.txt -r pipeline/requirements.txt

# Install spaCy model (CLI broken on Python 3.14 — use wheel directly)
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl

# 2. Run the pipeline (writes data/pipeline.db)
python -m pipeline.runner

# 3. Start the backend API  (terminal 1)
uvicorn api.main:app --host 0.0.0.0 --port 8000

# 4. Start the pipeline server  (terminal 2)
uvicorn pipeline.server:app --host 0.0.0.0 --port 8001

# 5. Start the frontend  (terminal 3)
cd frontend && npm install && npm run dev
```

Open **http://localhost:5173** — double-click nodes to expand, click edges to see source posts.

The **Text Analyzer** feature (type any sentence to extract causal relationships live) requires the pipeline server (step 4).

## Documentation

- [Architecture](docs/architecture.md) — system design with Mermaid diagrams
- [Getting Started](docs/getting-started.md) — detailed setup and configuration
- [Extending](docs/extending.md) — how to add new pipeline implementations
- [Graph Format](docs/graphformat.md) — SQLite schema contract between pipeline and API

## Structure

```
pipeline/           Python pipeline (4 pluggable steps) + pipeline REST server
api/                FastAPI backend (reads pipeline.db; no pipeline imports)
frontend/           React + Cytoscape.js UI
pipeline.yaml       Pipeline configuration — swap implementations without code changes
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

Edit `pipeline.yaml` and use the fully-qualified class name:

```yaml
step1_detection:
  implementation: "pipeline.step1_detection.regex_detector.RegexDetector"

step3_canonization:
  implementation: "pipeline.step3_canonization.transformer_canonizer.TransformerCanonizer"
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"

step4_hierarchy:
  implementation: "pipeline.step4_hierarchy.tfidf_clusterer.TFIDFClusterer"
```

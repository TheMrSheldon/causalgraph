# Getting Started

## Prerequisites

- Python ≥ 3.12
- Node.js ≥ 18 and npm ≥ 9
- The `rscience-submissions.parquet` file in the project root

> [!NOTE]
> This project was developed in a Python 3.14 dev container. If `hdbscan` or `spacy` fail to install (no wheels for your Python version), use Python 3.12 via `python3.12 -m venv .venv`.

---

## 1. Install Python dependencies

```bash
pip install -r api/requirements.txt -r pipeline/requirements.txt
pip install pytest pytest-asyncio ruff   # optional dev tools
```

spaCy's download CLI is broken on Python 3.14. Install the model wheel directly:

```bash
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl
```

On Python 3.12/3.13 the standard CLI works:

```bash
python -m spacy download en_core_web_sm
```

---

## 2. Run the pipeline

```bash
# Run all four steps with default config
python -m pipeline.runner

# Run only a specific step
python -m pipeline.runner --step 1   # detect causal posts
python -m pipeline.runner --step 2   # extract (cause, effect) pairs
python -m pipeline.runner --step 3   # canonize event descriptions
python -m pipeline.runner --step 4   # build cluster hierarchy

# Use a custom config file
python -m pipeline.runner --config my-pipeline.yaml
```

Pipeline output is written to `data/pipeline.db` (SQLite). You can inspect it:

```bash
sqlite3 data/pipeline.db "SELECT COUNT(*) FROM posts;"
sqlite3 data/pipeline.db "SELECT COUNT(*) FROM causal_relations;"
sqlite3 data/pipeline.db "SELECT level, COUNT(*) FROM clusters GROUP BY level;"
```

> [!TIP]
> Step 1 on 867K titles takes ~2–5 minutes with the regex detector. Steps 2–4 depend on the number of causal posts found (~10–20% of total).

---

## 3. Start the backend API

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

The backend reads `data/pipeline.db` (overridable via `GRAPH_DB_PATH` env var) and has **no dependency on the `pipeline/` package** — it is connected to the pipeline only via the SQLite database. See [docs/graphformat.md](graphformat.md) for the schema contract.

---

## 4. Start the pipeline server

The pipeline server exposes each pipeline step as a REST endpoint. It is used by the frontend's **Text Analyzer** feature (type any sentence to extract causal relationships live).

```bash
uvicorn pipeline.server:app --host 0.0.0.0 --port 8001
```

Endpoints (served under `/pipeline/` by default via `PIPELINE_ROOT` env var):

| Method | Path | Purpose |
|--------|------|---------|
| `POST /pipeline/detect` | `{"text": "..."}` | Step 1: classify text as causal or not |
| `POST /pipeline/extract` | `{"text": "..."}` | Steps 2+3: extract and canonize (cause, effect) pairs with spans |
| `GET /pipeline/health` | — | Liveness check |

The pipeline server uses the same `pipeline.yaml` implementations as the batch pipeline runner.

---

## 5. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser. The Vite dev server proxies:
- `/api/*` → backend API at `http://localhost:8000`
- `/pipeline/*` → pipeline server at `http://localhost:8001`

---

## Swapping pipeline components

Edit `pipeline.yaml` and change the `implementation` key to a fully-qualified class name:

```yaml
step3_canonization:
  implementation: "pipeline.step3_canonization.transformer_canonizer.TransformerCanonizer"
  model_name: "Qwen/Qwen2.5-1.5B-Instruct"
  batch_size: 8
  max_new_tokens: 24
  device: 0   # -1 = CPU, 0 = first CUDA GPU

step4_hierarchy:
  implementation: "pipeline.step4_hierarchy.embedding_ward_clusterer.EmbeddingWardClusterer"
  n_clusters_per_level: [500, 100, 30, 8]
```

Then re-run the affected steps:

```bash
python -m pipeline.runner --step 3
python -m pipeline.runner --step 4
```

No code changes required. The pipeline server picks up the new implementation on next restart.

---

## Available implementations

### Step 1 — Causality Detection

| Class | Description |
|-------|-------------|
| `pipeline.step1_detection.regex_detector.RegexDetector` | Compiled regex patterns *(default)* |

### Step 2 — Causal Extraction

| Class | Description |
|-------|-------------|
| `pipeline.step2_extraction.regex_spacy_extractor.RegexSpacyExtractor` | Regex patterns + spaCy dependency parse *(default)* |

### Step 3 — Canonization

| Class | Description |
|-------|-------------|
| `pipeline.step3_canonization.passthrough_canonizer.PassthroughCanonizer` | Return extraction spans as-is |
| `pipeline.step3_canonization.transformer_canonizer.TransformerCanonizer` | LLM rewrites spans into self-contained descriptions *(default)* |

### Step 4 — Hierarchy Inference

| Class | Description |
|-------|-------------|
| `pipeline.step4_hierarchy.tfidf_clusterer.TFIDFClusterer` | TF-IDF + Ward agglomerative clustering |
| `pipeline.step4_hierarchy.embedding_ward_clusterer.EmbeddingWardClusterer` | sentence-transformers + Ward linkage *(default)* |
| `pipeline.step4_hierarchy.embedding_clusterer.EmbeddingClusterer` | sentence-transformers + HDBSCAN |

---

## Running tests

```bash
pytest tests/ -v
```

> [!NOTE]
> `test_step2_extractor.py` requires spaCy's `en_core_web_sm` model (see installation above).

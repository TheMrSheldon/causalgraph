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
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
```

For LLM-based components (optional):

```bash
pip install -e ".[llm]"
export ANTHROPIC_API_KEY=your_key   # or OPENAI_API_KEY
```

---

## 2. Run the pipeline

```bash
# Run all four steps with default config
python scripts/run_pipeline.py

# Run only a specific step
python scripts/run_pipeline.py --step 1   # detect causal posts
python scripts/run_pipeline.py --step 2   # extract (cause, effect) pairs
python scripts/run_pipeline.py --step 3   # canonize event descriptions
python scripts/run_pipeline.py --step 4   # build cluster hierarchy

# Use a custom config file
python scripts/run_pipeline.py --config my-config.yaml
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

The backend reads `data/pipeline.db` and has **no dependency on the `pipeline/` package** — it is connected to the pipeline only via the SQLite database. See [docs/graphformat.md](graphformat.md) for the schema contract.

---

## 4. Start the pipeline server

The pipeline server exposes each pipeline step as a REST endpoint. It is used by the frontend's **Text Analyzer** feature (type any sentence to extract causal relations live).

```bash
uvicorn pipeline.server:app --host 0.0.0.0 --port 8001
```

Endpoints:

| Method | Path | Purpose |
|--------|------|---------|
| `POST /detect` | `{"text": "..."}` | Step 1: classify text as causal or not |
| `POST /extract` | `{"text": "..."}` | Steps 2+3: extract and canonize (cause, effect) pairs with spans |
| `GET /health` | — | Liveness check |

The pipeline server uses the same `config.yaml` implementations as the batch pipeline runner.

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

Edit `config.yaml` and change the `implementation` key for any step:

```yaml
pipeline:
  step1_detection:
    implementation: "zero_shot"   # was "regex"
    zero_shot_threshold: 0.80

  step3_canonization:
    implementation: "llm_anthropic"  # was "passthrough"

  step4_hierarchy:
    implementation: "embedding_ward"  # was "tfidf_ward"
    n_clusters_per_level: [500, 100, 30, 8]
```

Then re-run the affected steps:

```bash
python scripts/run_pipeline.py --step 1
python scripts/run_pipeline.py --step 3
python scripts/run_pipeline.py --step 4
```

No code changes required. The pipeline server picks up the new implementation on next restart.

---

## Available implementations

### Step 1 — Causality Detection

| Key | Description | Requirements |
|-----|-------------|--------------|
| `regex` | Compiled regex patterns *(default)* | None |
| `llm_openai` | OpenAI GPT batch classification | `OPENAI_API_KEY` |
| `llm_anthropic` | Anthropic Claude batch classification | `ANTHROPIC_API_KEY` |
| `zero_shot` | HuggingFace BART zero-shot NLI | `pip install .[zero-shot]` |

### Step 2 — Causal Extraction

| Key | Description | Requirements |
|-----|-------------|--------------|
| `regex_spacy` | Regex patterns + spaCy dep parse *(default)* | `en_core_web_sm` |
| `llm_openai` | OpenAI structured JSON extraction | `OPENAI_API_KEY` |
| `llm_anthropic` | Anthropic structured JSON extraction | `ANTHROPIC_API_KEY` |

### Step 3 — Canonization

| Key | Description | Requirements |
|-----|-------------|--------------|
| `passthrough` | Copy extraction spans as-is *(default)* | None |
| `llm_openai` | LLM rewrites spans into self-contained descriptions | `OPENAI_API_KEY` |
| `llm_anthropic` | LLM rewrites spans into self-contained descriptions | `ANTHROPIC_API_KEY` |

### Step 4 — Hierarchy Inference

| Key | Description | Requirements |
|-----|-------------|--------------|
| `tfidf_ward` | TF-IDF + Ward agglomerative clustering *(default)* | scikit-learn only |
| `embedding_ward` | sentence-transformers + Ward linkage | `sentence-transformers` |
| `embedding_hdbscan` | sentence-transformers + HDBSCAN | `pip install .[dev]` |
| `llm` | LLM-assigned topic labels | API key |

---

## Running tests

```bash
pytest tests/ -v
```

> [!NOTE]
> `test_step2_extractor.py` requires spaCy's `en_core_web_sm` model. Run `python -m spacy download en_core_web_sm` first.

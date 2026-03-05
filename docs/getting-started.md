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
# Run all three steps with default config
python scripts/run_pipeline.py

# Run only a specific step
python scripts/run_pipeline.py --step 1   # identify causal posts
python scripts/run_pipeline.py --step 2   # extract (cause, effect) pairs
python scripts/run_pipeline.py --step 3   # build cluster hierarchy

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
> Step 1 on 867K titles takes ~2–5 minutes with the regex identifier. Steps 2 and 3 depend on the number of causal posts found (~10–20% of total).

---

## 3. Start the API server

```bash
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

The API is available at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## 4. Start the frontend

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser.

---

## Swapping pipeline components

Edit `config.yaml` and change the `implementation` key for any step:

```yaml
pipeline:
  step1_identification:
    implementation: "zero_shot"   # was "regex"
    zero_shot_threshold: 0.80

  step3_hierarchy:
    implementation: "tfidf_ward"  # was "embedding_hdbscan"
    tfidf_n_top_clusters: 150
```

Then re-run the affected steps:

```bash
python scripts/run_pipeline.py --step 1
python scripts/run_pipeline.py --step 3
```

No code changes required.

---

## Available implementations

### Step 1 — Causality Identification

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

### Step 3 — Hierarchy Inference

| Key | Description | Requirements |
|-----|-------------|--------------|
| `embedding_hdbscan` | sentence-transformers + HDBSCAN *(default)* | `pip install .[dev]` |
| `tfidf_ward` | TF-IDF + Ward agglomerative clustering | scikit-learn only |
| `llm` | LLM-assigned topic labels | API key |

---

## Running tests

```bash
pytest tests/ -v
```

> [!NOTE]
> `test_step2_extractor.py` requires spaCy's `en_core_web_sm` model. Run `python -m spacy download en_core_web_sm` first.

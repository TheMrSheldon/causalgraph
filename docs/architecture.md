# Architecture

This document describes the architecture of the r/science causal relationship extraction and visualization pipeline.

## Overview

The project extracts causal claims from 867K Reddit r/science submission titles, structures them as a graph of events, and presents them in an interactive hierarchical visualization.

```mermaid
flowchart LR
    A[(rscience-submissions.parquet\n867K titles)] -->|PyArrow stream| B

    subgraph Pipeline["Pipeline (offline)"]
        B[Step 1\nCausality Detection] -->|causal posts| C
        C[Step 2\nCausal Extraction] -->|cause / effect pairs| D
        D[Step 3\nCanonicalization] -->|self-contained descriptions| E
        E[Step 4\nHierarchy Inference] -->|clustered graph| F
    end

    F[(pipeline.db\nSQLite)] -->|read-only| G
    F -->|schema contract| H

    subgraph Services["Runtime Services"]
        G[Backend API\nport 8000] -->|JSON| I[React + Cytoscape.js\nport 5173]
        H[Pipeline Server\nport 8001] -->|JSON| I
    end
```

Each pipeline step is **independently pluggable**: the implementation is selected at runtime via `pipeline.yaml` without any code changes.

The backend API (`api/`) and the pipeline (`pipeline/`) share **no Python imports** — they are connected only through the SQLite database. See [graphformat.md](graphformat.md) for the schema contract.

---

## Component Responsibilities

| Component | Responsibility | Key Files |
|-----------|----------------|-----------|
| `ParquetReader` | Stream 867K rows from Parquet in batches | `pipeline/parquet_reader.py` |
| `CausalityDetector` | Filter titles to those expressing causality | `pipeline/step1_detection/` |
| `CausalityExtractor` | Extract structured (cause, effect) pairs | `pipeline/step2_extraction/` |
| `EventCanonizer` | Rewrite event spans into self-contained descriptions | `pipeline/step3_canonization/` |
| `HierarchyInferrer` | Cluster events into multi-level hierarchy | `pipeline/step4_hierarchy/` |
| `Database` (pipeline) | SQLite DAL — schema, writes, graph queries | `pipeline/db.py` |
| `GraphDatabase` (API) | Read-only SQLite access for graph queries | `api/db.py` |
| Backend API | Serve graph data over REST | `api/` |
| Pipeline Server | Per-step REST API for live text analysis | `pipeline/server.py` |
| React + Cytoscape.js | Interactive hierarchical causal graph | `frontend/` |

---

## Data Flow

```mermaid
sequenceDiagram
    participant P as Parquet File
    participant S1 as Step 1<br/>Detector
    participant S2 as Step 2<br/>Extractor
    participant S3 as Step 3<br/>Canonizer
    participant S4 as Step 4<br/>Inferrer
    participant DB as SQLite DB
    participant API as Backend API
    participant PS as Pipeline Server
    participant UI as Browser

    P->>S1: batch of 5000 Post objects
    S1-->>DB: causal Posts (upsert)
    DB->>S2: all causal Posts
    S2-->>DB: CausalRelation rows
    DB->>S3: all CausalRelations
    S3-->>DB: canonical descriptions (UPDATE)
    DB->>S4: all CausalRelations (with canonical)
    S4-->>DB: EventCluster rows + memberships

    UI->>API: GET /api/graph
    API->>DB: get_clusters_at_level() + get_edges()
    API-->>UI: {nodes, edges}

    UI->>API: GET /api/clusters/{id}/expand
    API->>DB: get_children(id) + get_edges(child_ids)
    API-->>UI: {nodes, edges}

    UI->>API: GET /api/posts?source=&target=
    API->>DB: get_posts_for_edge()
    API-->>UI: {posts, total}

    UI->>PS: POST /extract {"text": "..."}
    PS-->>UI: {events, relations}
```

---

## Database Schema

See [graphformat.md](graphformat.md) for the full schema specification.

```mermaid
erDiagram
    posts {
        TEXT id PK
        TEXT title
        INTEGER score
        INTEGER num_comments
        INTEGER created_utc
        TEXT author
        TEXT url
        TEXT permalink
    }

    causal_relations {
        INTEGER id PK
        TEXT post_id FK
        TEXT cause_text
        TEXT effect_text
        TEXT cause_norm
        TEXT effect_norm
        TEXT cause_canonical
        TEXT effect_canonical
        REAL confidence
        TEXT extractor
        INTEGER is_countercausal
    }

    clusters {
        INTEGER id PK
        TEXT label
        INTEGER level
        INTEGER parent_id FK
        INTEGER member_count
        TEXT clusterer
    }

    cluster_members {
        INTEGER id PK
        INTEGER cluster_id FK
        INTEGER relation_id FK
        TEXT role
        TEXT event_text
    }

    leaf_edges {
        INTEGER source_cluster_id PK
        INTEGER target_cluster_id PK
        INTEGER relation_count
        INTEGER post_count
        REAL avg_score
        INTEGER countercausal_count
    }

    posts ||--o{ causal_relations : "has"
    causal_relations ||--o{ cluster_members : "referenced by"
    clusters ||--o{ cluster_members : "contains"
    clusters ||--o{ clusters : "parent of"
    clusters ||--o{ leaf_edges : "source"
    clusters ||--o{ leaf_edges : "target"
```

---

## Pluggable Interfaces

Each pipeline step is defined as a Python `Protocol` (structural typing). Swap implementations by changing one line in `pipeline.yaml`.

```mermaid
classDiagram
    class CausalityDetector {
        <<Protocol>>
        +detect(posts: list[Post]) list[Post]
        +name: str
    }

    class CausalityExtractor {
        <<Protocol>>
        +extract(post: Post) list[CausalRelation]
        +name: str
    }

    class EventCanonizer {
        <<Protocol>>
        +canonize(spans: list[tuple]) list[str]
        +name: str
    }

    class HierarchyInferrer {
        <<Protocol>>
        +infer(relations: list[CausalRelation]) tuple
        +name: str
    }

    CausalityDetector <|.. RegexDetector : implements

    CausalityExtractor <|.. RegexSpacyExtractor : implements

    EventCanonizer <|.. PassthroughCanonizer : implements
    EventCanonizer <|.. TransformerCanonizer : implements

    HierarchyInferrer <|.. EmbeddingWardClusterer : implements
    HierarchyInferrer <|.. TFIDFClusterer : implements
```

---

## Hierarchy Model

Events are organized into a multi-level tree. The number of levels is configured via `n_clusters_per_level` in `pipeline.yaml` — the list length determines the number of levels. The frontend renders the topmost level on load and supports drill-down via double-click.

```mermaid
graph TD
    T1["Level N — Top\n(10–50 nodes)\ne.g. 'cardiovascular / cancer'"]
    T2["Level N — Top\n'mental / cognitive'"]

    M1["Level 1 — Mid\n'heart / blood / pressure'"]
    M2["Level 1 — Mid\n'cancer / tumor / cell'"]
    M3["Level 1 — Mid\n'depression / anxiety / stress'"]

    L1["Level 0 — Leaf\n'blood pressure'"]
    L2["Level 0 — Leaf\n'heart rate'"]
    L3["Level 0 — Leaf\n'lung cancer'"]
    L4["Level 0 — Leaf\n'depression risk'"]

    T1 --> M1 --> L1
    M1 --> L2
    T1 --> M2 --> L3
    T2 --> M3 --> L4
```

Directed edges between clusters represent extracted causal relationships; edge weight encodes post count. The frontend always starts at the topmost level (`MAX(level)`) and infers this automatically from the API.

---

## API Reference

### Backend API (port 8000) — read-only graph endpoints

```mermaid
graph LR
    A[GET /api/graph] -->|min_post_count| B{Backend API}
    C[GET /api/graph/levels] --> B
    D[GET /api/clusters/id] --> B
    E[GET /api/clusters/id/expand] --> B
    F[GET /api/clusters/id/posts] --> B
    G[GET /api/posts] -->|source_cluster_id, target_cluster_id| B
    H[GET /api/posts/id] --> B
    B --> I[(SQLite DB)]
```

| Endpoint | Parameters | Purpose |
|----------|------------|---------|
| `GET /api/graph` | `min_post_count` | Top-level nodes + edges at the topmost hierarchy level |
| `GET /api/graph/levels` | — | Available levels and cluster counts |
| `GET /api/clusters/{id}` | — | Cluster detail: children, top events, sample posts |
| `GET /api/clusters/{id}/expand` | `min_post_count` | Child nodes + intra-cluster edges (drill-down) |
| `GET /api/clusters/{id}/posts` | `limit`, `offset`, `sort` | Paginated posts in cluster |
| `GET /api/posts` | `source_cluster_id`, `target_cluster_id` | Posts for a cause→effect edge click |
| `GET /api/posts/{id}` | — | Single post with extracted causal pair |

### Pipeline Server (port 8001) — live text analysis

| Endpoint | Body | Purpose |
|----------|------|---------|
| `POST /detect` | `{"text": "..."}` | Classify text as causal (`is_causal: bool`) |
| `POST /extract` | `{"text": "..."}` | Extract events and relations with character spans |
| `GET /health` | — | Liveness check |

---

## Frontend Interaction Model

```mermaid
stateDiagram-v2
    [*] --> TopLevel : page load\nGET /api/graph (topmost level)

    TopLevel --> Expanded : double-click node\nGET /api/clusters/id/expand
    Expanded --> TopLevel : right-click node\n(collapse)
    Expanded --> Expanded : double-click child\n(deeper expand)

    TopLevel --> EdgePosts : click edge\nGET /api/posts?source=&target=
    Expanded --> EdgePosts : click edge

    EdgePosts --> TopLevel : close drawer
    EdgePosts --> Expanded : close drawer
```

The Cytoscape.js graph uses **compound nodes** to represent expanded clusters. When a user double-clicks a cluster node:

1. `GET /api/clusters/{id}/expand` fetches child nodes and intra-cluster edges.
2. Child nodes are added to the Cytoscape element set with `parent: "cluster-{id}"`.
3. The parent node becomes a compound container; Cytoscape re-runs the layout.

Collapsing removes child elements and resets the parent to a regular node.

The graph layout algorithm can be changed in the Settings panel (gear icon). Available built-in options: `fcose`, `cose`, `breadthfirst`, `concentric`, `circle`.

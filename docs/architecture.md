# Architecture

This document describes the architecture of the r/science causal relationship extraction and visualization pipeline.

## Overview

The project extracts causal claims from 867K Reddit r/science submission titles, structures them as a graph of events, and presents them in an interactive hierarchical visualization.

```mermaid
flowchart LR
    A[(rscience-submissions.parquet\n867K titles)] -->|DuckDB stream| B

    subgraph Pipeline
        B[Step 1\nCausality Identification] -->|causal posts| C
        C[Step 2\nCausal Extraction] -->|cause / effect pairs| D
        D[Step 3\nHierarchy Inference] -->|clustered graph| E
    end

    E[(pipeline.db\nSQLite)] -->|REST API| F

    subgraph Frontend
        F[FastAPI] -->|JSON| G[React + Cytoscape.js]
    end
```

Each pipeline step is **independently pluggable**: the implementation is selected at runtime via `config.yaml` without any code changes.

---

## Component Responsibilities

| Component | Responsibility | Key Files |
|-----------|----------------|-----------|
| `ParquetReader` | Stream 867K rows from Parquet in batches | `pipeline/parquet_reader.py` |
| `CausalityIdentifier` | Filter titles to those expressing causality | `pipeline/step1_identification/` |
| `CausalExtractor` | Extract structured (cause, effect) pairs | `pipeline/step2_extraction/` |
| `HierarchyInferrer` | Cluster events into 3-level hierarchy | `pipeline/step3_hierarchy/` |
| `Database` | SQLite DAL — schema, writes, graph queries | `pipeline/db.py` |
| `Registry` | Load implementations from `config.yaml` | `pipeline/registry.py` |
| FastAPI app | Serve graph data over REST | `api/` |
| React + Cytoscape.js | Interactive hierarchical causal graph | `frontend/` |

---

## Data Flow

```mermaid
sequenceDiagram
    participant P as Parquet File
    participant S1 as Step 1<br/>Identifier
    participant S2 as Step 2<br/>Extractor
    participant S3 as Step 3<br/>Inferrer
    participant DB as SQLite DB
    participant API as FastAPI
    participant UI as Browser

    P->>S1: batch of 5000 Post objects
    S1-->>DB: causal Posts (upsert)
    DB->>S2: all causal Posts
    S2-->>DB: CausalRelation rows
    DB->>S3: all CausalRelations
    S3-->>DB: EventCluster rows + memberships
    UI->>API: GET /api/graph?level=2
    API->>DB: get_clusters_at_level(2) + get_edges()
    API-->>UI: {nodes, edges}
    UI->>API: GET /api/clusters/{id}/expand
    API->>DB: get_children(id) + get_edges(child_ids)
    API-->>UI: {nodes, edges}
    UI->>API: GET /api/posts?source=&target=
    API->>DB: get_posts_for_edge()
    API-->>UI: {posts, total}
```

---

## Database Schema

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
        REAL confidence
        TEXT extractor
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

    posts ||--o{ causal_relations : "has"
    causal_relations ||--o{ cluster_members : "referenced by"
    clusters ||--o{ cluster_members : "contains"
    clusters ||--o{ clusters : "parent of"
```

The `graph_edges` VIEW joins `cluster_members` to produce aggregated cause→effect edges between clusters, used by all graph API endpoints.

---

## Pluggable Interfaces

Each pipeline step is defined as a Python `Protocol` (structural typing). Swap implementations by changing one line in `config.yaml`.

```mermaid
classDiagram
    class CausalityIdentifier {
        <<Protocol>>
        +identify(posts: list[Post]) list[Post]
        +name: str
    }

    class CausalExtractor {
        <<Protocol>>
        +extract(post: Post) list[CausalRelation]
        +name: str
    }

    class HierarchyInferrer {
        <<Protocol>>
        +infer(relations: list[CausalRelation]) tuple
        +name: str
    }

    CausalityIdentifier <|.. RegexIdentifier : implements
    CausalityIdentifier <|.. LLMIdentifier : implements
    CausalityIdentifier <|.. ZeroShotIdentifier : implements

    CausalExtractor <|.. RegexSpacyExtractor : implements
    CausalExtractor <|.. LLMExtractor : implements

    HierarchyInferrer <|.. EmbeddingClusterer : implements
    HierarchyInferrer <|.. TFIDFClusterer : implements
    HierarchyInferrer <|.. LLMTopicGrouper : implements
```

---

## Hierarchy Model

Events are organized into a 3-level tree. The frontend renders each level as Cytoscape.js compound nodes.

```mermaid
graph TD
    T1["Level 2 — Top\n(20–50 nodes)\ne.g. 'cardiovascular / cancer'"]
    T2["Level 2 — Top\n'mental / cognitive'"]

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

Directed edges between clusters represent extracted causal relationships; edge weight encodes post count.

---

## API Reference

All endpoints are read-only (`GET`). The API is served at `http://localhost:8000`.

```mermaid
graph LR
    A[GET /api/graph] -->|level, min_post_count| B{FastAPI}
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
| `GET /api/graph` | `level`, `min_post_count` | Top-level nodes + edges at a given hierarchy level |
| `GET /api/graph/levels` | — | Available levels and cluster counts |
| `GET /api/clusters/{id}` | — | Cluster detail: children, top events, sample posts |
| `GET /api/clusters/{id}/expand` | `min_post_count` | Child nodes + intra-cluster edges (drill-down) |
| `GET /api/clusters/{id}/posts` | `limit`, `offset`, `sort` | Paginated posts in cluster |
| `GET /api/posts` | `source_cluster_id`, `target_cluster_id` | Posts for a cause→effect edge click |
| `GET /api/posts/{id}` | — | Single post with extracted causal pair |

---

## Frontend Interaction Model

```mermaid
stateDiagram-v2
    [*] --> TopLevel : page load\nGET /api/graph?level=2

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
3. The parent node becomes a compound container; Cytoscape re-runs the `fcose` layout.

Collapsing removes child elements and resets the parent to a regular node.

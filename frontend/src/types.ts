export interface ClusterNode {
  id: number;
  label: string;
  level: number; // 0=leaf, 1=mid, 2=top
  parent_id: number | null;
  member_count: number;
}

export interface GraphEdge {
  source_cluster_id: number;
  target_cluster_id: number;
  relation_count: number;
  post_count: number;
  avg_score: number;
  countercausal_count: number;
}

export interface GraphData {
  nodes: ClusterNode[];
  edges: GraphEdge[];
}

export interface PostSummary {
  id: string;
  title: string;
  score: number;
  num_comments: number;
  created_utc: number;
  permalink: string | null;
}

export interface PostDetail extends PostSummary {
  cause_text: string | null;
  effect_text: string | null;
  confidence: number | null;
}

export interface ClusterDetail {
  cluster: ClusterNode;
  children: ClusterNode[];
  top_events: string[];
  posts: EdgePostSummary[];
}

export interface PaginatedPosts {
  posts: PostSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface RelationSpan {
  cause_text: string;
  effect_text: string;
  cause_canonical: string | null;
  effect_canonical: string | null;
  cause_cluster_id: number | null;
  effect_cluster_id: number | null;
  relation_type: 'causal' | 'countercausal' | 'no_rel';
}

export interface EdgePostSummary extends PostSummary {
  relations: RelationSpan[];
}

export interface PaginatedEdgePosts {
  posts: EdgePostSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface LevelCounts {
  levels: number[];
  counts: Record<string, number>;
}

export interface SelectedEdge {
  source_cluster_id: number;
  target_cluster_id: number;
}

// --- Path Finder -------------------------------------------------------

export interface PathNode {
  id: string
  label: string
  level: number
  type: 'source' | 'intermediate' | 'target'
}

export interface PathLink {
  source: string  // PathNode.id
  target: string  // PathNode.id
  post_count: number
}

export interface PathsResponse {
  nodes: PathNode[]
  links: PathLink[]
}

// --- Text Analyzer -----------------------------------------------------

/** A unique event identified in the text, with its span position. */
export interface AnalysisEvent {
  index: number        // palette index; same event text → same color
  span_text: string    // text as it appears in the original document
  description: string  // cleaned event phrase (may differ for LLM extractors)
  start: number
  end: number
}

/** One causal or countercausal relationship with per-label certainty scores. */
export interface AnalysisRelation {
  cause_event_index: number
  effect_event_index: number
  cause_text: string
  effect_text: string
  cause_canonical: string   // self-contained description from Step 3; may equal cause_text
  effect_canonical: string  // self-contained description from Step 3; may equal effect_text
  relation_type: 'causal' | 'countercausal' | 'no_rel'
  p_none: number
  p_causal: number
  p_countercausal: number
}

export interface AnalysisResponse {
  text: string
  events: AnalysisEvent[]
  relations: AnalysisRelation[]
}

// --- Shared ------------------------------------------------------------

export type VisualizationMode = 'no' | 'opacity' | 'size'
export type NodeSpacing = 'tight' | 'normal' | 'spread'
export type LayoutAlgorithm = 'fcose' | 'cose' | 'breadthfirst' | 'circle' | 'concentric'

export type AnimationSpeed = 'off' | 'fast' | 'normal' | 'slow'

export interface GraphSettings {
  clusterSizeMode: VisualizationMode
  linkSizeMode: VisualizationMode
  showEdgeLabels: boolean
  showMemberCount: boolean
  dimOnSelection: boolean
  highlightOnHover: boolean
  nodeSpacing: NodeSpacing
  layoutAlgorithm: LayoutAlgorithm
  animationSpeed: AnimationSpeed
  showArrows: boolean
  showLegend: boolean
  showHighlightSpans: boolean
}

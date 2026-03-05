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
  posts: PostSummary[];
}

export interface PaginatedPosts {
  posts: PostSummary[];
  total: number;
  limit: number;
  offset: number;
}

export interface EdgePostSummary extends PostSummary {
  cause_text: string | null;
  effect_text: string | null;
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

export interface AnalysisSpan {
  start: number
  end: number
  type: 'cause' | 'effect'
}

export interface AnalysisRelation {
  cause_text: string
  effect_text: string
  cause_cluster_id: number | null
  cause_cluster_label: string | null
  effect_cluster_id: number | null
  effect_cluster_label: string | null
  corpus_post_count: number
}

export interface AnalysisResponse {
  text: string
  spans: AnalysisSpan[]
  relations: AnalysisRelation[]
}

// --- Shared ------------------------------------------------------------

export type VisualizationMode = 'no' | 'opacity' | 'size'
export type NodeSpacing = 'tight' | 'normal' | 'spread'

export interface GraphSettings {
  clusterSizeMode: VisualizationMode
  linkSizeMode: VisualizationMode
  showEdgeLabels: boolean
  showMemberCount: boolean
  dimOnSelection: boolean
  highlightOnHover: boolean
  nodeSpacing: NodeSpacing
  animateLayout: boolean
  showArrows: boolean
  showLegend: boolean
}

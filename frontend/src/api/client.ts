import axios from 'axios'
import type {
  AnalysisResponse,
  AnalysisSpan,
  ClusterDetail,
  GraphData,
  LevelCounts,
  PaginatedEdgePosts,
  PaginatedPosts,
  PathsResponse,
  PostDetail,
} from '../types'

// ---------------------------------------------------------------------------
// Mock helpers (used until backend endpoints are implemented)
// ---------------------------------------------------------------------------

function mockPaths(causeQuery: string, effectQuery: string): PathsResponse {
  return {
    nodes: [
      { id: 'src', label: causeQuery, level: 2, type: 'source' },
      { id: 'mid0', label: 'Physiological mechanisms', level: 1, type: 'intermediate' },
      { id: 'mid1', label: 'Behavioural pathways', level: 1, type: 'intermediate' },
      { id: 'tgt', label: effectQuery, level: 2, type: 'target' },
    ],
    links: [
      { source: 'src', target: 'mid0', post_count: 312 },
      { source: 'src', target: 'mid1', post_count: 178 },
      { source: 'mid0', target: 'tgt', post_count: 287 },
      { source: 'mid1', target: 'tgt', post_count: 154 },
    ],
  }
}

const CAUSAL_VERB_RE =
  /\b(causes?|leads?\s+to|results?\s+in|increases?|reduces?|improves?|decreases?|boosts?|prevents?|promotes?|triggers?)\b/gi

function mockAnalyze(text: string): AnalysisResponse {
  const spans: AnalysisSpan[] = []
  const relations: AnalysisResponse['relations'] = []
  const re = new RegExp(CAUSAL_VERB_RE.source, 'gi')
  let m: RegExpExecArray | null
  while ((m = re.exec(text)) !== null) {
    const verbIdx = m.index
    const verbEnd = verbIdx + m[0].length
    // cause = text since last period/start of string
    const sentStart = Math.max(0, text.lastIndexOf('.', verbIdx - 1) + 1)
    const causeText = text.slice(sentStart, verbIdx).replace(/^\s+/, '')
    // effect = text until next punctuation/end
    const rest = text.slice(verbEnd).replace(/^\s+/, '')
    const effMatch = rest.match(/^([^.,;]+)/)
    const effectText = effMatch ? effMatch[1].trim() : ''
    if (!causeText || !effectText) continue
    const causeStart = text.indexOf(causeText, sentStart)
    if (causeStart !== -1) spans.push({ start: causeStart, end: causeStart + causeText.length, type: 'cause' })
    const effStart = verbEnd + (text.slice(verbEnd).length - rest.length)
    spans.push({ start: effStart, end: effStart + effectText.length, type: 'effect' })
    relations.push({ cause_text: causeText, effect_text: effectText,
      cause_cluster_id: null, cause_cluster_label: null,
      effect_cluster_id: null, effect_cluster_label: null,
      corpus_post_count: 0 })
  }
  return { text, spans, relations }
}

const http = axios.create({ baseURL: '/api' })

export const api = {
  getGraph: async (level = 2, minPostCount = 1): Promise<GraphData> => {
    const { data } = await http.get<GraphData>('/graph', {
      params: { level, min_post_count: minPostCount },
    })
    return data
  },

  getLevels: async (): Promise<LevelCounts> => {
    const { data } = await http.get<LevelCounts>('/graph/levels')
    return data
  },

  getCluster: async (id: number): Promise<ClusterDetail> => {
    const { data } = await http.get<ClusterDetail>(`/clusters/${id}`)
    return data
  },

  expandCluster: async (id: number, contextIds: number[] = []): Promise<GraphData> => {
    const { data } = await http.get<GraphData>(`/clusters/${id}/expand`, {
      params: contextIds.length ? { context_ids: contextIds.join(',') } : undefined,
    })
    return data
  },

  getClusterPosts: async (
    id: number,
    limit = 50,
    offset = 0,
    sort = 'score'
  ): Promise<PaginatedPosts> => {
    const { data } = await http.get<PaginatedPosts>(`/clusters/${id}/posts`, {
      params: { limit, offset, sort },
    })
    return data
  },

  getPostsForEdge: async (
    sourceClusterId: number,
    targetClusterId: number,
    limit = 50,
    offset = 0
  ): Promise<PaginatedEdgePosts> => {
    const { data } = await http.get<PaginatedEdgePosts>('/posts', {
      params: {
        source_cluster_id: sourceClusterId,
        target_cluster_id: targetClusterId,
        limit,
        offset,
      },
    })
    return data
  },

  getPost: async (id: string): Promise<PostDetail> => {
    const { data } = await http.get<PostDetail>(`/posts/${id}`)
    return data
  },

  // TODO: implement GET /api/paths?cause_query=&effect_query=&min_posts=1&max_hops=2
  findPaths: async (causeQuery: string, effectQuery: string): Promise<PathsResponse> => {
    return mockPaths(causeQuery, effectQuery)
  },

  // TODO: implement POST /api/analyze  { text: string } → AnalysisResponse
  analyzeText: async (text: string): Promise<AnalysisResponse> => {
    return mockAnalyze(text)
  },
}

import axios from 'axios'
import type {
  AnalysisResponse,
  ClusterDetail,
  GraphData,
  LevelCounts,
  PaginatedEdgePosts,
  PaginatedPosts,
  PathsResponse,
  PostDetail,
} from '../types'

// ---------------------------------------------------------------------------
// Custom endpoint overrides (persisted in localStorage)
// ---------------------------------------------------------------------------

const _ENDPOINTS_KEY = 'api-endpoints'

let _backendOverride = ''
let _pipelineOverride = ''

// Load saved overrides at module initialisation
try {
  const raw = localStorage.getItem(_ENDPOINTS_KEY)
  if (raw) {
    const p = JSON.parse(raw)
    _backendOverride  = p.backendUrl  ?? ''
    _pipelineOverride = p.pipelineUrl ?? ''
  }
} catch { /* ignore parse errors */ }

/**
 * Update the axios base URLs used by all subsequent API calls.
 * Pass empty strings to revert to the default proxied paths.
 */
export function setApiOverrides(backendUrl: string, pipelineUrl: string): void {
  _backendOverride  = backendUrl
  _pipelineOverride = pipelineUrl
}

/** Returns the currently active overrides (may be empty strings = proxy default). */
export function getApiOverrides(): { backendUrl: string; pipelineUrl: string } {
  return { backendUrl: _backendOverride, pipelineUrl: _pipelineOverride }
}

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


// ---------------------------------------------------------------------------
// Axios instances with intercept-based URL overrides
// ---------------------------------------------------------------------------

const http = axios.create({ baseURL: '/api' })

// When a custom backend URL is set, rewrite the request URL to the full path
// so the browser sends it cross-origin instead of through the local proxy.
//   e.g.  baseURL='/api' + url='/graph'  →  'http://host:8000/api/graph'
http.interceptors.request.use((config) => {
  if (_backendOverride && config.url) {
    const path = config.url.startsWith('/') ? config.url : `/${config.url}`
    config.url     = `${_backendOverride.replace(/\/$/, '')}/api${path}`
    config.baseURL = ''
  }
  return config
})

const pipelineHttp = axios.create({ baseURL: '/pipeline' })

// Pipeline override: the user provides the full base including any path prefix,
// so we just prepend it directly to the existing path.
//   e.g.  url='/extract'  →  'http://host:8001/extract'
//         url='/extract'  →  'http://host/pipeline/extract'  (via reverse proxy)
pipelineHttp.interceptors.request.use((config) => {
  if (_pipelineOverride && config.url) {
    const path = config.url.startsWith('/') ? config.url : `/${config.url}`
    config.url     = `${_pipelineOverride.replace(/\/$/, '')}${path}`
    config.baseURL = ''
  }
  return config
})

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

  analyzeText: async (text: string): Promise<AnalysisResponse> => {
    const { data } = await pipelineHttp.post<AnalysisResponse>('/extract', { text })
    return data
  },
}

import axios from 'axios'
import type {
  ClusterDetail,
  GraphData,
  LevelCounts,
  PaginatedPosts,
  PostDetail,
} from '../types'

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

  expandCluster: async (id: number): Promise<GraphData> => {
    const { data } = await http.get<GraphData>(`/clusters/${id}/expand`)
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
  ): Promise<PaginatedPosts> => {
    const { data } = await http.get<PaginatedPosts>('/posts', {
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
}

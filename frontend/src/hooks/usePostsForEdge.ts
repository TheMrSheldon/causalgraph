import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'
import type { SelectedEdge } from '../types'

export function usePostsForEdge(edge: SelectedEdge | null, limit = 50) {
  return useQuery({
    queryKey: ['posts-for-edge', edge?.source_cluster_id, edge?.target_cluster_id, limit],
    queryFn: () => api.getPostsForEdge(edge!.source_cluster_id, edge!.target_cluster_id, limit),
    enabled: edge !== null,
    staleTime: 5 * 60 * 1000,
  })
}

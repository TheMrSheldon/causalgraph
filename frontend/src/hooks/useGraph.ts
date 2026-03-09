import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

export function useGraph(level: number, minPostCount = 1) {
  return useQuery({
    queryKey: ['graph', level, minPostCount],
    queryFn: () => api.getGraph(level, minPostCount),
    staleTime: 5 * 60 * 1000,
  })
}

export function useLevels() {
  return useQuery({
    queryKey: ['levels'],
    queryFn: api.getLevels,
    staleTime: Infinity,
  })
}

export function useCluster(id: number | null) {
  return useQuery({
    queryKey: ['cluster', id],
    queryFn: () => api.getCluster(id!),
    enabled: id !== null,
    staleTime: 5 * 60 * 1000,
  })
}

export function useClusterPosts(id: number | null, limit = 50) {
  return useQuery({
    queryKey: ['cluster-posts', id, limit],
    queryFn: () => api.getClusterPosts(id!, limit),
    enabled: id !== null,
    staleTime: 5 * 60 * 1000,
  })
}

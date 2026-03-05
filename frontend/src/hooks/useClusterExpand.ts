import { useCallback, useState } from 'react'
import { api } from '../api/client'
import type { ClusterNode, GraphEdge } from '../types'

interface ExpandState {
  nodes: ClusterNode[]
  edges: GraphEdge[]
}

export function useClusterExpand() {
  const [expandedClusters, setExpandedClusters] = useState<Set<number>>(new Set())
  const [cache, setCache] = useState<Map<number, ExpandState>>(new Map())
  const [loading, setLoading] = useState(false)

  const expandCluster = useCallback(
    async (clusterId: number, contextIds: number[] = []) => {
      if (!cache.has(clusterId)) {
        setLoading(true)
        try {
          const data = await api.expandCluster(clusterId, contextIds)
          setCache((prev) => new Map(prev).set(clusterId, data))
        } finally {
          setLoading(false)
        }
      }
      setExpandedClusters((prev) => new Set(prev).add(clusterId))
    },
    [cache]
  )

  const collapseCluster = useCallback((clusterId: number) => {
    setExpandedClusters((prev) => {
      const next = new Set(prev)
      next.delete(clusterId)
      return next
    })
  }, [])

  const isExpanded = useCallback(
    (clusterId: number) => expandedClusters.has(clusterId),
    [expandedClusters]
  )

  const getExpandedData = useCallback(
    (clusterId: number): ExpandState | undefined => cache.get(clusterId),
    [cache]
  )

  return { expandCluster, collapseCluster, isExpanded, getExpandedData, loading }
}

import { useCallback, useState } from 'react'
import { api } from '../api/client'
import type { ClusterNode, GraphEdge } from '../types'

interface ExpandState {
  nodes: ClusterNode[]
  edges: GraphEdge[]
}

export function useClusterExpand(minPostCount = 1) {
  const [expandedClusters, setExpandedClusters] = useState<Set<number>>(new Set())
  // Cache key: `${clusterId}:${minPostCount}` — re-fetches when threshold changes
  const [cache, setCache] = useState<Map<string, ExpandState>>(new Map())
  const [loading, setLoading] = useState(false)

  const expandCluster = useCallback(
    async (clusterId: number, contextIds: number[] = []) => {
      const key = `${clusterId}:${minPostCount}`
      if (!cache.has(key)) {
        setLoading(true)
        try {
          const data = await api.expandCluster(clusterId, contextIds, minPostCount)
          setCache((prev) => new Map(prev).set(key, data))
        } finally {
          setLoading(false)
        }
      }
      setExpandedClusters((prev) => new Set(prev).add(clusterId))
    },
    [cache, minPostCount]
  )

  const collapseCluster = useCallback((clusterId: number) => {
    setExpandedClusters((prev) => {
      const next = new Set(prev)
      // BFS: collapse the node and every expanded descendant at any depth
      const queue = [clusterId]
      while (queue.length > 0) {
        const id = queue.shift()!
        next.delete(id)
        const key = `${id}:${minPostCount}`
        const children = cache.get(key)
        if (children) {
          for (const child of children.nodes) {
            if (next.has(child.id)) queue.push(child.id)
          }
        }
      }
      return next
    })
  }, [cache, minPostCount])

  const isExpanded = useCallback(
    (clusterId: number) => expandedClusters.has(clusterId),
    [expandedClusters]
  )

  const getExpandedData = useCallback(
    (clusterId: number): ExpandState | undefined => cache.get(`${clusterId}:${minPostCount}`),
    [cache, minPostCount]
  )

  return { expandCluster, collapseCluster, isExpanded, getExpandedData, expandedClusters, loading }
}

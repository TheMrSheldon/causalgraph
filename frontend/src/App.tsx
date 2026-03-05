import { useCallback, useState } from 'react'
import { CausalGraph } from './components/CausalGraph'
import { ClusterPanel } from './components/ClusterPanel'
import { FilterBar } from './components/FilterBar'
import { PostList } from './components/PostList'
import { useClusterExpand } from './hooks/useClusterExpand'
import { useGraph } from './hooks/useGraph'
import './styles/graph.css'
import type { SelectedEdge } from './types'

export default function App() {
  const [level, setLevel] = useState(2)
  const [minPostCount, setMinPostCount] = useState(1)
  const [selectedCluster, setSelectedCluster] = useState<number | null>(null)
  const [selectedEdge, setSelectedEdge] = useState<SelectedEdge | null>(null)

  const { data: graphData, isLoading } = useGraph(level, minPostCount)
  const { expandCluster, collapseCluster, isExpanded, getExpandedData } = useClusterExpand()

  const childNodesByParent = (() => {
    const map = new Map()
    for (const nodeId of Array.from({ length: 10000 }, (_, i) => i)) {
      const d = getExpandedData(nodeId)
      if (d) map.set(nodeId, d.nodes)
    }
    return map
  })()

  const childEdgesByParent = (() => {
    const map = new Map()
    for (const nodeId of Array.from({ length: 10000 }, (_, i) => i)) {
      const d = getExpandedData(nodeId)
      if (d) map.set(nodeId, d.edges)
    }
    return map
  })()

  const expandedNodes = new Set(
    Array.from({ length: 10000 }, (_, i) => i).filter(isExpanded)
  )

  const handleNodeDblClick = useCallback(
    (clusterId: number, level: number) => {
      if (level > 0) {
        if (isExpanded(clusterId)) {
          collapseCluster(clusterId)
        } else {
          expandCluster(clusterId)
        }
      }
    },
    [isExpanded, expandCluster, collapseCluster]
  )

  const handleNodeRightClick = useCallback(
    (clusterId: number) => {
      collapseCluster(clusterId)
    },
    [collapseCluster]
  )

  const handleNodeClick = useCallback((clusterId: number) => {
    setSelectedCluster(clusterId)
    setSelectedEdge(null)
  }, [])

  const handleEdgeClick = useCallback((edge: SelectedEdge) => {
    setSelectedEdge(edge)
    setSelectedCluster(null)
  }, [])

  const handleExpandRequest = useCallback(
    (clusterId: number, _level: number) => {
      expandCluster(clusterId)
    },
    [expandCluster]
  )

  return (
    <div className="app-layout">
      <header className="app-header">
        <h1>r/science Causal Graph</h1>
        <FilterBar
          level={level}
          minPostCount={minPostCount}
          onLevelChange={setLevel}
          onMinPostCountChange={setMinPostCount}
        />
        {isLoading && <span style={{ fontSize: 12, color: '#64748b' }}>Loading…</span>}
      </header>

      <main className="graph-container">
        {graphData && (
          <CausalGraph
            nodes={graphData.nodes}
            edges={graphData.edges}
            expandedNodes={expandedNodes}
            childNodesByParent={childNodesByParent}
            childEdgesByParent={childEdgesByParent}
            onNodeDblClick={handleNodeDblClick}
            onNodeRightClick={handleNodeRightClick}
            onNodeClick={handleNodeClick}
            onEdgeClick={handleEdgeClick}
          />
        )}
        {selectedEdge && (
          <PostList edge={selectedEdge} onClose={() => setSelectedEdge(null)} />
        )}
      </main>

      <aside className="sidebar">
        <ClusterPanel
          clusterId={selectedCluster}
          onExpandRequest={handleExpandRequest}
        />
      </aside>
    </div>
  )
}

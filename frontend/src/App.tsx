import { useCallback, useMemo, useState } from 'react'
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

  const childNodesByParent = useMemo(() => {
    const map = new Map()
    for (const nodeId of Array.from({ length: 10000 }, (_, i) => i)) {
      const d = getExpandedData(nodeId)
      if (d) map.set(nodeId, d.nodes)
    }
    return map
  }, [getExpandedData])

  const childEdgesByParent = useMemo(() => {
    const map = new Map()
    for (const nodeId of Array.from({ length: 10000 }, (_, i) => i)) {
      const d = getExpandedData(nodeId)
      if (d) map.set(nodeId, d.edges)
    }
    return map
  }, [getExpandedData])

  const expandedNodes = useMemo(
    () => new Set(Array.from({ length: 10000 }, (_, i) => i).filter(isExpanded)),
    [isExpanded]
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
        <a className="app-header-brand" href="https://webis.de/" target="_blank" rel="noopener noreferrer">
          <img src="/webis-logo.png" alt="Webis" />
          <span>Webis.de</span>
        </a>
        <div className="app-header-divider" />
        <h1>r/science Causal Graph</h1>
        <div className="app-header-spacer" />
        <FilterBar
          level={level}
          minPostCount={minPostCount}
          onLevelChange={setLevel}
          onMinPostCountChange={setMinPostCount}
        />
        {isLoading && <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)' }}>Loading…</span>}
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

      <footer className="app-footer">
        <span>
          © 2026{' '}
          <a href="https://webis.de/" target="_blank" rel="noopener noreferrer">Webis Group</a>
        </span>
        <span className="footer-sep">•</span>
        <a href="https://github.com/webis-de" target="_blank" rel="noopener noreferrer" title="GitHub">
          <svg width="14" height="14" viewBox="0 0 20 20" fill="currentColor"><path d="M10,1 C5.03,1 1,5.03 1,10 C1,13.98 3.58,17.35 7.16,18.54 C7.61,18.62 7.77,18.34 7.77,18.11 C7.77,17.9 7.76,17.33 7.76,16.58 C5.26,17.12 4.73,15.37 4.73,15.37 C4.32,14.33 3.73,14.05 3.73,14.05 C2.91,13.5 3.79,13.5 3.79,13.5 C4.69,13.56 5.17,14.43 5.17,14.43 C5.97,15.8 7.28,15.41 7.79,15.18 C7.87,14.6 8.1,14.2 8.36,13.98 C6.36,13.75 4.26,12.98 4.26,9.53 C4.26,8.55 4.61,7.74 5.19,7.11 C5.1,6.88 4.79,5.97 5.28,4.73 C5.28,4.73 6.04,4.49 7.75,5.65 C8.47,5.45 9.24,5.35 10,5.35 C10.76,5.35 11.53,5.45 12.25,5.65 C13.97,4.48 14.72,4.73 14.72,4.73 C15.21,5.97 14.9,6.88 14.81,7.11 C15.39,7.74 15.73,8.54 15.73,9.53 C15.73,12.99 13.63,13.75 11.62,13.97 C11.94,14.25 12.23,14.8 12.23,15.64 C12.23,16.84 12.22,17.81 12.22,18.11 C12.22,18.35 12.38,18.63 12.84,18.54 C16.42,17.35 19,13.98 19,10 C19,5.03 14.97,1 10,1 Z"/></svg>
        </a>
        <span className="footer-sep">•</span>
        <a href="https://webis.de/people.html" target="_blank" rel="noopener noreferrer">Contact</a>
        <span className="footer-sep">•</span>
        <a href="https://webis.de/legal.html" target="_blank" rel="noopener noreferrer">Impressum / Terms / Privacy</a>
      </footer>
    </div>
  )
}

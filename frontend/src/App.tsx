import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { CausalGraph } from './components/CausalGraph'
import { ClusterPanel } from './components/ClusterPanel'
import { PathFinderScreen } from './components/PathFinderScreen'
import { PostList } from './components/PostList'
import { SettingsModal } from './components/SettingsModal'
import { StatsModal } from './components/StatsModal'
import { TextAnalyzerScreen } from './components/TextAnalyzerScreen'
import { useClusterExpand } from './hooks/useClusterExpand'
import { useGraph, useLevels } from './hooks/useGraph'
import { readUrlState, syncUrlState } from './hooks/useUrlSync'
import type { Screen } from './hooks/useUrlSync'
import { getApiOverrides, setApiOverrides } from './api/client'
import './styles/graph.css'
import type { GraphSettings, SelectedEdge } from './types'

// Parse URL state once at module load time so useState initialisers can use it
const initialUrl = readUrlState()

export default function App() {
  const [activeScreen, setActiveScreen] = useState<Screen>(initialUrl.screen)
  const [minPostCount, setMinPostCount] = useState(1)
  const [selectedCluster, setSelectedCluster] = useState<number | null>(initialUrl.node)
  const [selectedEdge, setSelectedEdge] = useState<SelectedEdge | null>(
    initialUrl.edge
      ? { source_cluster_id: initialUrl.edge.source, target_cluster_id: initialUrl.edge.target }
      : null
  )
  const [highlightedPostId, setHighlightedPostId] = useState<string | null>(initialUrl.post)
  const [focusStackIds, setFocusStackIds] = useState<number[]>(initialUrl.focus)
  const [sidebarWidth, setSidebarWidth] = useState(340)
  const dragState = useRef<{ startX: number; startWidth: number } | null>(null)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [statsOpen, setStatsOpen] = useState(false)

  // Backend / pipeline override URLs (reactive so they're synced to URL)
  const [backendUrl,  setBackendUrl]  = useState(() => {
    // URL params win over localStorage on first load
    const { backend, pipeline } = initialUrl
    if (backend || pipeline) setApiOverrides(backend, pipeline)
    return getApiOverrides().backendUrl
  })
  const [pipelineUrl, setPipelineUrl] = useState(() => getApiOverrides().pipelineUrl)

  const [settings, setSettings] = useState<GraphSettings>(() => {
    const defaults: GraphSettings = {
      clusterSizeMode: 'no',
      linkSizeMode: 'size',
      showEdgeLabels: false,
      showMemberCount: false,
      dimOnSelection: true,
      highlightOnHover: false,
      nodeSpacing: 'normal',
      layoutAlgorithm: 'fcose',
      animationSpeed: 'normal' as const,
      showArrows: true,
      showLegend: true,
      showHighlightSpans: true,
    }
    try {
      const saved = localStorage.getItem('graph-settings')
      return saved ? { ...defaults, ...JSON.parse(saved) } : defaults
    } catch {
      return defaults
    }
  })

  useEffect(() => {
    try { localStorage.setItem('graph-settings', JSON.stringify(settings)) } catch { /* ignore */ }
  }, [settings])

  useEffect(() => {
    const onMouseMove = (e: MouseEvent) => {
      if (!dragState.current) return
      const delta = dragState.current.startX - e.clientX
      setSidebarWidth(Math.max(200, Math.min(700, dragState.current.startWidth + delta)))
    }
    const onMouseUp = () => { dragState.current = null }
    document.addEventListener('mousemove', onMouseMove)
    document.addEventListener('mouseup', onMouseUp)
    return () => {
      document.removeEventListener('mousemove', onMouseMove)
      document.removeEventListener('mouseup', onMouseUp)
    }
  }, [])

  const { data: levelsData } = useLevels()
  const topLevel = levelsData ? Math.max(...levelsData.levels) : 2
  const { data: graphData, isLoading } = useGraph(topLevel, minPostCount)
  const { expandCluster, collapseCluster, isExpanded, getExpandedData } = useClusterExpand()

  // Restore expanded nodes from URL on first graph load
  const pendingExpansionsRef = useRef<number[]>(initialUrl.expanded)
  useEffect(() => {
    if (!graphData || pendingExpansionsRef.current.length === 0) return
    const contextIds = graphData.nodes.map((n) => n.id)
    for (const id of pendingExpansionsRef.current) {
      if (graphData.nodes.some((n) => n.id === id)) {
        expandCluster(id, contextIds.filter((cid) => cid !== id))
      }
    }
    pendingExpansionsRef.current = []
  }, [graphData, expandCluster])

  // Sync all navigable state → URL (replaceState so back/forward isn't polluted)
  useEffect(() => {
    syncUrlState({
      screen:   activeScreen,
      node:     selectedCluster,
      edge:     selectedEdge
        ? { source: selectedEdge.source_cluster_id, target: selectedEdge.target_cluster_id }
        : null,
      expanded: Array.from(expandedNodes),
      focus:    focusStackIds,
      post:     highlightedPostId,
      backend:  backendUrl,
      pipeline: pipelineUrl,
    })
  // expandedNodes is derived below — its identity changes on every expansion;
  // include it by spreading the dependency list manually
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeScreen, selectedCluster, selectedEdge, focusStackIds, highlightedPostId, backendUrl, pipelineUrl])

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

  // Sync expandedNodes to URL separately (its deps aren't easy to list above)
  useEffect(() => {
    syncUrlState({
      screen:   activeScreen,
      node:     selectedCluster,
      edge:     selectedEdge
        ? { source: selectedEdge.source_cluster_id, target: selectedEdge.target_cluster_id }
        : null,
      expanded: Array.from(expandedNodes),
      focus:    focusStackIds,
      post:     highlightedPostId,
      backend:  backendUrl,
      pipeline: pipelineUrl,
    })
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [expandedNodes])

  const clusterLabels = useMemo(() => {
    const map = new Map<number, string>()
    for (const node of graphData?.nodes ?? []) map.set(node.id, node.label)
    for (const nodes of childNodesByParent.values()) {
      for (const node of nodes) map.set(node.id, node.label)
    }
    return map
  }, [graphData, childNodesByParent])

  const handleNodeDblClick = useCallback(
    (clusterId: number, level: number) => {
      if (level > 0) {
        if (isExpanded(clusterId)) {
          collapseCluster(clusterId)
        } else {
          const contextIds: number[] = []
          for (const node of graphData?.nodes ?? []) {
            if (node.id !== clusterId) contextIds.push(node.id)
          }
          for (const [parentId, nodes] of childNodesByParent.entries()) {
            if (isExpanded(parentId)) {
              for (const n of nodes) if (n.id !== clusterId) contextIds.push(n.id)
            }
          }
          expandCluster(clusterId, contextIds)
        }
      }
    },
    [isExpanded, expandCluster, collapseCluster, graphData, childNodesByParent]
  )

  const handleNodeRightClick = useCallback(
    (clusterId: number) => { collapseCluster(clusterId) },
    [collapseCluster]
  )

  const handleNodeClick = useCallback((clusterId: number) => {
    setSelectedCluster(clusterId)
    setSelectedEdge(null)
    setHighlightedPostId(null)
  }, [])

  const handleEdgeClick = useCallback((edge: SelectedEdge) => {
    setSelectedEdge(edge)
    setSelectedCluster(null)
    setHighlightedPostId(null)
  }, [])

  const handleDeselect = useCallback(() => {
    setSelectedCluster(null)
    setSelectedEdge(null)
    setHighlightedPostId(null)
  }, [])

  const handleClusterClick = useCallback((clusterId: number) => {
    setSelectedCluster(clusterId)
    setSelectedEdge(null)
    setHighlightedPostId(null)
  }, [])

  const isExplorer = activeScreen === 'explorer'

  return (
    <div
      className={`app-layout${isExplorer ? '' : ' no-sidebar'}`}
      style={isExplorer ? { '--sidebar-width': `${sidebarWidth}px` } as React.CSSProperties : undefined}
    >
      <header className="app-header">
        <a className="app-header-brand" href="https://webis.de/" target="_blank" rel="noopener noreferrer">
          <img src="/webis-logo.png" alt="Webis" />
          <span>Webis.de</span>
        </a>
        <div className="app-header-divider" />
        <h1>r/science Causal Graph</h1>
        <div className="app-header-spacer" />
        <>
          {isExplorer && isLoading && <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)' }}>Loading…</span>}
          <button className="header-icon-btn" onClick={() => setStatsOpen(true)} title="Statistics" disabled={!graphData}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <line x1="18" y1="20" x2="18" y2="10"/>
              <line x1="12" y1="20" x2="12" y2="4"/>
              <line x1="6" y1="20" x2="6" y2="14"/>
            </svg>
          </button>
          <button className="header-icon-btn" onClick={() => setSettingsOpen(true)} title="Settings">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3"/>
              <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06A1.65 1.65 0 0 0 19.4 9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"/>
            </svg>
          </button>
        </>
      </header>

      <nav className="app-tabs">
        {([
          ['explorer',   'Graph Explorer'],
          ['pathfinder', 'Path Finder'],
          ['analyzer',   'Text Analyzer'],
        ] as [Screen, string][]).map(([id, label]) => (
          <button
            key={id}
            className={`app-tab${activeScreen === id ? ' app-tab--active' : ''}`}
            onClick={() => setActiveScreen(id)}
          >
            {label}
          </button>
        ))}
      </nav>

      <StatsModal
        open={statsOpen}
        onClose={() => setStatsOpen(false)}
        nodes={graphData?.nodes ?? []}
        edges={graphData?.edges ?? []}
        clusterLabels={clusterLabels}
      />

      <SettingsModal
        open={settingsOpen}
        settings={settings}
        onSettingsChange={setSettings}
        onClose={() => setSettingsOpen(false)}
        minPostCount={minPostCount}
        onMinPostCountChange={setMinPostCount}
        activeScreen={activeScreen}
        onEndpointsChange={(backend, pipeline) => {
          setBackendUrl(backend)
          setPipelineUrl(pipeline)
        }}
      />

      {isExplorer && (
        <>
          <main className="graph-container">
            {graphData && (
              <CausalGraph
                nodes={graphData.nodes}
                edges={graphData.edges}
                expandedNodes={expandedNodes}
                childNodesByParent={childNodesByParent}
                childEdgesByParent={childEdgesByParent}
                settings={settings}
                selectedClusterId={selectedCluster}
                onNodeDblClick={handleNodeDblClick}
                onNodeRightClick={handleNodeRightClick}
                onNodeClick={handleNodeClick}
                onEdgeClick={handleEdgeClick}
                onDeselect={handleDeselect}
                onFocusStackChange={setFocusStackIds}
                initialFocusIds={initialUrl.focus.length > 0 ? initialUrl.focus : undefined}
              />
            )}
          </main>

          <aside className="sidebar">
            <div
              className="sidebar-resize-handle"
              onMouseDown={(e) => {
                dragState.current = { startX: e.clientX, startWidth: sidebarWidth }
                e.preventDefault()
              }}
            />
            {selectedEdge
              ? <PostList
                  edge={selectedEdge}
                  sourceLabel={clusterLabels.get(selectedEdge.source_cluster_id)}
                  targetLabel={clusterLabels.get(selectedEdge.target_cluster_id)}
                  onClusterClick={handleClusterClick}
                  highlightedPostId={highlightedPostId}
                />
              : <ClusterPanel
                  clusterId={selectedCluster}
                  clusterLabels={clusterLabels}
                  onClusterClick={handleClusterClick}
                  isExpanded={isExpanded}
                  onCollapseRequest={collapseCluster}
                  highlightedPostId={highlightedPostId}
                />
            }
          </aside>
        </>
      )}

      {activeScreen === 'pathfinder' && (
        <div className="screen-fullwidth">
          <PathFinderScreen />
        </div>
      )}

      {activeScreen === 'analyzer' && (
        <div className="screen-fullwidth">
          <TextAnalyzerScreen />
        </div>
      )}

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

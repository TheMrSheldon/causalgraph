import cytoscape, { type ElementDefinition, type StylesheetStyle } from 'cytoscape'
// @ts-expect-error – no types for fcose
import fcose from 'cytoscape-fcose'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { ClusterNode, GraphEdge, GraphSettings, NodeSpacing, SelectedEdge } from '../types'

cytoscape.use(fcose)

const SPACING_PARAMS: Record<NodeSpacing, { nodeRepulsion: number; idealEdgeLength: number; nodeSeparation: number }> = {
  tight:  { nodeRepulsion: 150000, idealEdgeLength: 200, nodeSeparation: 150 },
  normal: { nodeRepulsion: 300000, idealEdgeLength: 350, nodeSeparation: 250 },
  spread: { nodeRepulsion: 600000, idealEdgeLength: 500, nodeSeparation: 400 },
}

function buildFcoseLayout(nodeSpacing: NodeSpacing, animate: boolean): cytoscape.LayoutOptions {
  const sp = SPACING_PARAMS[nodeSpacing]
  return {
    name: 'fcose',
    animate,
    animationDuration: 600,
    nodeRepulsion: () => sp.nodeRepulsion,
    idealEdgeLength: () => sp.idealEdgeLength,
    nodeSeparation: sp.nodeSeparation,
    gravity: 0.04,
    gravityRange: 1.5,
    padding: 60,
    numIter: 5000,
  } as cytoscape.LayoutOptions
}

// Webis graph colour palette — mirrors lecturenotes table tier colours
// Canvas bg: #ffffff  (white, uk-section-default)
// Level-2 (top / table header):  #929292  lecturenotes table header color
// Level-1 (mid / part row):      #b8b8b8  lighter step
// Level-0 (leaf / unit row):     #d4d4d4  lightest — with dark text
// Selected:                      #f0506e  Webis danger red
// Edges:                         #aaaaaa  neutral mid-gray

export const LEVEL_COLORS = {
  2: '#929292',
  1: '#b8b8b8',
  0: '#d4d4d4',
} as const

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type AnyStyle = any

function buildCytoscapeStyle(settings: GraphSettings): StylesheetStyle[] {
  const { clusterSizeMode, linkSizeMode, showEdgeLabels, showMemberCount, showArrows } = settings

  // Node size per level — fixed or mapped to member_count
  const nodeSize = (fixed: number, mapRange: [number, number]): AnyStyle =>
    clusterSizeMode === 'size'
      ? { width: `mapData(member_count, 1, 5000, ${mapRange[0]}, ${mapRange[1]})`,
          height: `mapData(member_count, 1, 5000, ${mapRange[0]}, ${mapRange[1]})` }
      : { width: fixed, height: fixed }

  const nodeOpacity: AnyStyle = clusterSizeMode === 'opacity'
    ? { opacity: 'mapData(member_count, 1, 5000, 0.3, 1)' }
    : {}

  // Edge width and opacity
  const edgeWidth: AnyStyle = linkSizeMode === 'no'
    ? { width: 1.5 }
    : linkSizeMode === 'size'
      ? { width: 'mapData(post_count, 1, 500, 1, 10)' }
      : { width: 2 }

  const edgeOpacity: AnyStyle = linkSizeMode === 'opacity'
    ? { opacity: 'mapData(post_count, 1, 500, 0.15, 1)' }
    : { opacity: 1 }

  // Highlighted edge width (always visible, slightly bolder)
  const hlWidth: AnyStyle = linkSizeMode === 'size'
    ? { width: 'mapData(post_count, 1, 500, 2, 14)' }
    : { width: 3 }

  const edgeLabelStyle: AnyStyle = showEdgeLabels ? {
    label: 'data(post_count)',
    'font-size': '9px',
    color: '#666',
    'text-background-color': '#fff',
    'text-background-opacity': 0.85,
    'text-background-shape': 'roundrectangle',
    'text-background-padding': '2px',
  } : { label: '' }

  return [
    {
      selector: 'node',
      style: {
        label: showMemberCount ? 'data(label_with_count)' : 'data(label)',
        'text-valign': 'bottom',
        'text-halign': 'center',
        'text-margin-y': 5,
        'text-wrap': 'wrap',
        'text-max-width': '110px',
        'font-size': '11px',
        'font-weight': '600',
        'font-family': '"Noto Sans", Verdana, sans-serif',
        color: '#222',
        'text-outline-width': 0,
        'text-background-color': '#fff',
        'text-background-opacity': 0.88,
        'text-background-shape': 'roundrectangle',
        'text-background-padding': '2px',
        'border-width': 0,
        ...nodeOpacity,
      },
    },
    {
      selector: 'node.level-2',
      style: {
        'background-color': LEVEL_COLORS[2],
        ...nodeSize(52, [28, 90]),
        'font-size': '12px',
      },
    },
    {
      selector: 'node.level-1',
      style: {
        'background-color': LEVEL_COLORS[1],
        ...nodeSize(38, [20, 65]),
      },
    },
    {
      selector: 'node.level-0',
      style: {
        'background-color': LEVEL_COLORS[0],
        ...nodeSize(26, [14, 45]),
        'font-size': '10px',
      },
    },
    {
      selector: ':compound',
      style: {
        'background-opacity': 0.06,
        'background-color': '#aaaaaa',
        'border-width': 1,
        'border-color': '#aaaaaa',
        'border-opacity': 0.35,
        padding: '22px',
      },
    },
    {
      selector: 'node:selected',
      style: {
        'border-width': 3,
        'border-color': '#f0506e',
      },
    },
    {
      selector: 'edge',
      style: {
        'line-color': '#aaaaaa',
        'target-arrow-color': '#aaaaaa',
        'target-arrow-shape': showArrows ? 'triangle' : 'none',
        'curve-style': 'bezier',
        ...edgeWidth,
        ...edgeOpacity,
        ...edgeLabelStyle,
      },
    },
    {
      selector: 'edge:selected',
      style: {
        'line-color': '#f0506e',
        'target-arrow-color': '#f0506e',
        opacity: 1,
      },
    },
    {
      selector: 'edge.edge-outgoing',
      style: {
        'line-color': '#1e87f0',
        'target-arrow-color': '#1e87f0',
        opacity: 1,
        ...hlWidth,
      },
    },
    {
      selector: 'edge.edge-incoming',
      style: {
        'line-color': '#faa05a',
        'target-arrow-color': '#faa05a',
        opacity: 1,
        ...hlWidth,
      },
    },
    {
      selector: '.dimmed',
      style: { opacity: 0.15 },
    },
  ]
}

interface CausalGraphProps {
  nodes: ClusterNode[]
  edges: GraphEdge[]
  expandedNodes: Set<number>
  childNodesByParent: Map<number, ClusterNode[]>
  childEdgesByParent: Map<number, GraphEdge[]>
  settings: GraphSettings
  onNodeDblClick: (clusterId: number, level: number) => void
  onNodeRightClick: (clusterId: number) => void
  onNodeClick: (clusterId: number) => void
  onEdgeClick: (edge: SelectedEdge) => void
}

function toCytoscapeElements(
  nodes: ClusterNode[],
  edges: GraphEdge[],
  expandedNodes: Set<number>,
  childNodesByParent: Map<number, ClusterNode[]>,
  childEdgesByParent: Map<number, GraphEdge[]>
): ElementDefinition[] {
  const elements: ElementDefinition[] = []
  const allNodes: ClusterNode[] = [...nodes]
  const allEdges: GraphEdge[] = [...edges]

  // Inject child nodes and edges for each expanded cluster
  for (const parentId of expandedNodes) {
    const children = childNodesByParent.get(parentId)
    if (children) allNodes.push(...children)
    const childEdges = childEdgesByParent.get(parentId)
    if (childEdges) allEdges.push(...childEdges)
  }

  const nodeIdSet = new Set(allNodes.map((n) => `cluster-${n.id}`))

  for (const node of allNodes) {
    const parentId =
      node.parent_id !== null && expandedNodes.has(node.parent_id)
        ? `cluster-${node.parent_id}`
        : undefined

    elements.push({
      data: {
        id: `cluster-${node.id}`,
        label: node.label,
        label_with_count: `${node.label}\n${node.member_count.toLocaleString()}`,
        level: node.level,
        member_count: node.member_count,
        ...(parentId ? { parent: parentId } : {}),
      },
      classes: `level-${node.level}`,
    })
  }

  // Deduplicate edges (same pair may appear from top-level and expanded)
  const edgeSet = new Set<string>()
  for (const edge of allEdges) {
    const key = `${edge.source_cluster_id}-${edge.target_cluster_id}`
    if (edgeSet.has(key)) continue
    edgeSet.add(key)

    const srcId = `cluster-${edge.source_cluster_id}`
    const tgtId = `cluster-${edge.target_cluster_id}`
    if (!nodeIdSet.has(srcId) || !nodeIdSet.has(tgtId)) continue

    elements.push({
      data: {
        id: `edge-${edge.source_cluster_id}-${edge.target_cluster_id}`,
        source: srcId,
        target: tgtId,
        post_count: edge.post_count,
        relation_count: edge.relation_count,
        avg_score: edge.avg_score,
      },
    })
  }

  return elements
}

export function CausalGraph({
  nodes,
  edges,
  expandedNodes,
  childNodesByParent,
  childEdgesByParent,
  settings,
  onNodeDblClick,
  onNodeRightClick,
  onNodeClick,
  onEdgeClick,
}: CausalGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<cytoscape.Core | null>(null)
  const settingsRef = useRef(settings)
  const selectedNodeRef = useRef<number | null>(null)
  const layoutInitRef = useRef(false)
  const [contextMenu, setContextMenu] = useState<{
    x: number; y: number; clusterId: number; level: number
  } | null>(null)

  // Keep settingsRef current
  useEffect(() => { settingsRef.current = settings }, [settings])

  // Initialize Cytoscape once
  useEffect(() => {
    if (!containerRef.current) return

    cyRef.current = cytoscape({
      container: containerRef.current,
      style: buildCytoscapeStyle(settings),
      layout: buildFcoseLayout(settings.nodeSpacing, settings.animateLayout),
      userZoomingEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: false,
    })

    return () => {
      cyRef.current?.destroy()
      cyRef.current = null
    }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // Wire event handlers
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return

    cy.removeAllListeners()

    const applyHighlight = (node: cytoscape.NodeSingular) => {
      cy.elements().removeClass('edge-outgoing edge-incoming dimmed')
      const outgoing = node.outgoers('edge')
      const incoming = node.incomers('edge')
      const neighborNodes = node.neighborhood('node')
      const kept = outgoing.union(incoming).union(neighborNodes).union(node)
      if (settingsRef.current.dimOnSelection) cy.elements().not(kept).addClass('dimmed')
      outgoing.addClass('edge-outgoing')
      incoming.addClass('edge-incoming')
    }

    cy.on('dblclick', 'node', (evt) => {
      const id = parseInt(evt.target.id().replace('cluster-', ''), 10)
      const level = evt.target.data('level') as number
      onNodeDblClick(id, level)
    })

    cy.on('cxttap', 'node', (evt) => {
      evt.originalEvent.preventDefault()
      const id = parseInt(evt.target.id().replace('cluster-', ''), 10)
      const level = evt.target.data('level') as number
      const { clientX, clientY } = evt.originalEvent as MouseEvent
      setContextMenu({ x: clientX, y: clientY, clusterId: id, level })
    })

    cy.on('tap', 'node', (evt) => {
      const node = evt.target
      const id = parseInt(node.id().replace('cluster-', ''), 10)
      selectedNodeRef.current = id
      applyHighlight(node)
      onNodeClick(id)
    })

    cy.on('mouseover', 'node', (evt) => {
      if (!settingsRef.current.highlightOnHover) return
      applyHighlight(evt.target)
    })

    cy.on('mouseout', 'node', () => {
      if (!settingsRef.current.highlightOnHover) return
      cy.elements().removeClass('edge-outgoing edge-incoming dimmed')
      if (selectedNodeRef.current !== null) {
        const sel = cy.$(`#cluster-${selectedNodeRef.current}`)
        if (sel.length > 0) applyHighlight(sel)
      }
    })

    cy.on('tap', 'edge', (evt) => {
      cy.elements().removeClass('edge-outgoing edge-incoming dimmed')
      selectedNodeRef.current = null
      const parts = evt.target.id().split('-')
      const src = parseInt(parts[1], 10)
      const tgt = parseInt(parts[2], 10)
      onEdgeClick({ source_cluster_id: src, target_cluster_id: tgt })
    })

    cy.on('tap', (evt) => {
      if (evt.target === cy) {
        selectedNodeRef.current = null
        cy.elements().removeClass('edge-outgoing edge-incoming dimmed')
      }
    })
  }, [onNodeDblClick, onNodeRightClick, onNodeClick, onEdgeClick])

  // Re-apply style when settings change (no layout re-run)
  useEffect(() => {
    cyRef.current?.style(buildCytoscapeStyle(settings))
  }, [settings])

  // Re-run layout when spacing or animation settings change
  useEffect(() => {
    if (!layoutInitRef.current) { layoutInitRef.current = true; return }
    cyRef.current?.layout(buildFcoseLayout(settings.nodeSpacing, settings.animateLayout)).run()
  }, [settings.nodeSpacing, settings.animateLayout]) // eslint-disable-line react-hooks/exhaustive-deps

  // Update elements when data changes
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return

    const elements = toCytoscapeElements(
      nodes,
      edges,
      expandedNodes,
      childNodesByParent,
      childEdgesByParent
    )

    cy.style(buildCytoscapeStyle(settingsRef.current))
    cy.elements().removeClass('edge-outgoing edge-incoming dimmed').remove()
    cy.add(elements)
    cy.layout(buildFcoseLayout(settingsRef.current.nodeSpacing, settingsRef.current.animateLayout)).run()
  }, [nodes, edges, expandedNodes, childNodesByParent, childEdgesByParent]) // eslint-disable-line react-hooks/exhaustive-deps

  // Dismiss context menu on outside click
  useEffect(() => {
    if (!contextMenu) return
    const dismiss = () => setContextMenu(null)
    document.addEventListener('click', dismiss)
    document.addEventListener('contextmenu', dismiss)
    return () => {
      document.removeEventListener('click', dismiss)
      document.removeEventListener('contextmenu', dismiss)
    }
  }, [contextMenu])

  const isExpanded = expandedNodes.has(contextMenu?.clusterId ?? -1)
  const canExpand = (contextMenu?.level ?? 0) > 0 && !isExpanded
  const canCollapse = isExpanded

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <div ref={containerRef} style={{ width: '100%', height: '100%' }} />

      {contextMenu && (
        <div
          className="node-context-menu"
          style={{ left: contextMenu.x, top: contextMenu.y }}
          onClick={(e) => e.stopPropagation()}
        >
          {canExpand && (
            <button className="node-context-item" onClick={() => {
              onNodeDblClick(contextMenu.clusterId, contextMenu.level)
              setContextMenu(null)
            }}>
              <svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5"><polyline points="7 4 13 10 7 16"/></svg>
              Expand sub-clusters
            </button>
          )}
          {canCollapse && (
            <button className="node-context-item" onClick={() => {
              onNodeRightClick(contextMenu.clusterId)
              setContextMenu(null)
            }}>
              <svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5"><polyline points="13 4 7 10 13 16"/></svg>
              Collapse sub-clusters
            </button>
          )}
          {!canExpand && !canCollapse && (
            <span className="node-context-empty">No actions available</span>
          )}
        </div>
      )}

      {settings.showLegend && (
        <div className="graph-legend">
          <div className="graph-legend-item">
            <span className="graph-legend-dot" style={{ background: LEVEL_COLORS[2] }} />
            Top cluster
          </div>
          <div className="graph-legend-item">
            <span className="graph-legend-dot" style={{ background: LEVEL_COLORS[1] }} />
            Mid cluster
          </div>
          <div className="graph-legend-item">
            <span className="graph-legend-dot" style={{ background: LEVEL_COLORS[0] }} />
            Leaf cluster
          </div>
          <div className="graph-legend-hint">
            Double-click to expand<br />
            Right-click for menu<br />
            Click edge for posts
          </div>
        </div>
      )}
    </div>
  )
}

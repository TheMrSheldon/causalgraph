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
    {
      selector: 'node.focus-dimmed',
      style: { opacity: 0.06, events: 'no' } as AnyStyle,
    },
    {
      selector: 'edge.focus-dimmed',
      style: { opacity: 0 },
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
  selectedClusterId?: number | null
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
  // Expanded parents are removed from the graph — replaced by their children
  const allNodes: ClusterNode[] = nodes.filter((n) => !expandedNodes.has(n.id))
  const allEdges: GraphEdge[] = [...edges]

  // Add child nodes (free-floating, no parent reference) and edges
  for (const parentId of expandedNodes) {
    const children = childNodesByParent.get(parentId)
    // Skip children that are themselves expanded (replaced by their own children)
    if (children) allNodes.push(...children.filter((c) => !expandedNodes.has(c.id)))
    const childEdges = childEdgesByParent.get(parentId)
    if (childEdges) allEdges.push(...childEdges)
  }

  const nodeIdSet = new Set(allNodes.map((n) => `cluster-${n.id}`))

  for (const node of allNodes) {
    elements.push({
      data: {
        id: `cluster-${node.id}`,
        label: node.label,
        label_with_count: `${node.label}\n${node.member_count.toLocaleString()}`,
        level: node.level,
        member_count: node.member_count,
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
        countercausal_count: edge.countercausal_count ?? 0,
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
  selectedClusterId,
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
  const prevNodesRef = useRef<ClusterNode[]>([])
  const prevEdgesRef = useRef<GraphEdge[]>([])
  const [contextMenu, setContextMenu] = useState<{
    x: number; y: number; clusterId: number; level: number; label: string
  } | null>(null)
  const [focusStack, setFocusStack] = useState<Array<{ id: number; label: string; level: number }>>([])
  const focusStackRef = useRef(focusStack)
  useEffect(() => { focusStackRef.current = focusStack }, [focusStack])
  // Nodes that were expanded automatically on focus entry (so we can collapse on exit)
  const focusAutoExpandedRef = useRef<Set<number>>(new Set())

  // Keep refs current
  useEffect(() => { settingsRef.current = settings }, [settings])
  const childNodesByParentRef = useRef(childNodesByParent)
  useEffect(() => { childNodesByParentRef.current = childNodesByParent }, [childNodesByParent])

  // Highlight a single node's edges
  const applyHighlight = useCallback((node: cytoscape.NodeSingular) => {
    const cy = cyRef.current
    if (!cy) return
    cy.elements().removeClass('edge-outgoing edge-incoming dimmed')
    const outgoing = node.outgoers('edge')
    const incoming = node.incomers('edge')
    const neighborNodes = node.neighborhood('node')
    const kept = outgoing.union(incoming).union(neighborNodes).union(node)
    if (settingsRef.current.dimOnSelection) cy.elements().not(kept).addClass('dimmed')
    outgoing.addClass('edge-outgoing')
    incoming.addClass('edge-incoming')
  }, [])

  // Apply focus: ghost non-children nodes, hide non-children edges, zoom to focused subset
  const applyFocus = useCallback((focusedId: number, animate = true) => {
    const cy = cyRef.current
    if (!cy) return
    const children = childNodesByParentRef.current.get(focusedId)
    const childIds: Set<string> = children && children.length > 0
      ? new Set(children.map((c) => `cluster-${c.id}`))
      : new Set([`cluster-${focusedId}`])
    cy.nodes().forEach((n) => {
      if (childIds.has(n.id())) n.removeClass('focus-dimmed')
      else n.addClass('focus-dimmed')
    })
    cy.edges().forEach((e) => {
      const inFocus = childIds.has(e.source().id()) && childIds.has(e.target().id())
      if (inFocus) e.removeClass('focus-dimmed')
      else e.addClass('focus-dimmed')
    })
    const focusedEles = cy.nodes().filter((n) => childIds.has(n.id()))
    if (focusedEles.length > 0) {
      cy.animate({ fit: { eles: focusedEles, padding: 80 } as cytoscape.Fit, duration: animate ? 400 : 0 })
    }
  }, [])

  const clearFocus = useCallback((animate = true) => {
    const cy = cyRef.current
    if (!cy) return
    cy.elements().removeClass('focus-dimmed')
    cy.animate({ fit: { eles: cy.elements(), padding: 60 } as cytoscape.Fit, duration: animate ? 400 : 0 })
  }, [])

  // Pop one focus level; collapse the node if it was auto-expanded by focus entry
  const exitFocus = useCallback(() => {
    const stack = focusStackRef.current
    if (stack.length === 0) return
    const last = stack[stack.length - 1]
    if (focusAutoExpandedRef.current.has(last.id)) {
      focusAutoExpandedRef.current.delete(last.id)
      onNodeRightClick(last.id)
    }
    setFocusStack((s) => s.slice(0, -1))
  }, [onNodeRightClick])

  // Clear entire focus stack; collapse all auto-expanded nodes
  const clearAllFocus = useCallback(() => {
    for (const id of focusAutoExpandedRef.current) {
      onNodeRightClick(id)
    }
    focusAutoExpandedRef.current.clear()
    setFocusStack([])
  }, [onNodeRightClick])

  // Re-apply current focus whenever the stack changes (with animation)
  useEffect(() => {
    if (focusStack.length === 0) clearFocus(true)
    else applyFocus(focusStack[focusStack.length - 1].id, true)
  }, [focusStack, applyFocus, clearFocus])

  // Highlight all children of an expanded parent (parent not in graph)
  const applyChildrenHighlight = useCallback((parentId: number) => {
    const cy = cyRef.current
    if (!cy) return
    const children = childNodesByParentRef.current.get(parentId)
    if (!children) return
    const childIdSet = new Set(children.map((c) => `cluster-${c.id}`))
    const childNodes = cy.nodes().filter((n) => childIdSet.has(n.id()))
    if (childNodes.length === 0) return
    cy.elements().unselect().removeClass('edge-outgoing edge-incoming dimmed')
    childNodes.select()
    const outgoing = childNodes.outgoers('edge')
    const incoming = childNodes.incomers('edge')
    const kept = outgoing.union(incoming).union(childNodes.neighborhood('node')).union(childNodes)
    if (settingsRef.current.dimOnSelection) cy.elements().not(kept).addClass('dimmed')
    outgoing.addClass('edge-outgoing')
    incoming.addClass('edge-incoming')
  }, [])

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
      zoomingFactor: 0.05,   // each scroll step changes zoom by 5% (near-linear feel)
      minZoom: 0.1,
      maxZoom: 5,
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

    cy.on('dblclick', 'node', (evt) => {
      const id = parseInt(evt.target.id().replace('cluster-', ''), 10)
      const level = evt.target.data('level') as number
      onNodeDblClick(id, level)
    })

    cy.on('cxttap', 'node', (evt) => {
      evt.originalEvent.preventDefault()
      const id = parseInt(evt.target.id().replace('cluster-', ''), 10)
      const level = evt.target.data('level') as number
      const label = evt.target.data('label') as string
      const { clientX, clientY } = evt.originalEvent as MouseEvent
      setContextMenu({ x: clientX, y: clientY, clusterId: id, level, label })
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
        else applyChildrenHighlight(selectedNodeRef.current)
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
        cy.elements().unselect().removeClass('edge-outgoing edge-incoming dimmed')
      }
    })
  }, [applyHighlight, applyChildrenHighlight, onNodeDblClick, onNodeRightClick, onNodeClick, onEdgeClick])

  // Re-apply style when settings change (no layout re-run)
  useEffect(() => {
    cyRef.current?.style(buildCytoscapeStyle(settings))
  }, [settings])

  // Re-run layout when spacing or animation settings change
  useEffect(() => {
    if (!layoutInitRef.current) { layoutInitRef.current = true; return }
    cyRef.current?.layout(buildFcoseLayout(settings.nodeSpacing, settings.animateLayout)).run()
  }, [settings.nodeSpacing, settings.animateLayout]) // eslint-disable-line react-hooks/exhaustive-deps

  // Programmatically highlight node when selectedClusterId changes (from sidebar links)
  useEffect(() => {
    if (selectedClusterId == null) return
    const cy = cyRef.current
    if (!cy) return
    selectedNodeRef.current = selectedClusterId
    const node = cy.$(`#cluster-${selectedClusterId}`)
    if (node.length > 0) {
      cy.elements().unselect()
      node.select()
      applyHighlight(node)
    } else {
      // Parent is expanded — highlight its children instead
      applyChildrenHighlight(selectedClusterId)
    }
  }, [selectedClusterId, applyHighlight, applyChildrenHighlight])

  // Update elements when data changes — incremental on expand/collapse, full rebuild on new graph data
  useEffect(() => {
    const cy = cyRef.current
    if (!cy) return

    const isFullRebuild =
      cy.elements().length === 0 ||
      prevNodesRef.current !== nodes ||
      prevEdgesRef.current !== edges

    prevNodesRef.current = nodes
    prevEdgesRef.current = edges

    const elements = toCytoscapeElements(
      nodes,
      edges,
      expandedNodes,
      childNodesByParent,
      childEdgesByParent
    )

    if (isFullRebuild) {
      cy.style(buildCytoscapeStyle(settingsRef.current))
      cy.elements().removeClass('edge-outgoing edge-incoming dimmed').remove()
      cy.add(elements)
      cy.layout(buildFcoseLayout(settingsRef.current.nodeSpacing, settingsRef.current.animateLayout)).run()
      return
    }

    // Incremental diff: remove stale elements, add new ones
    const targetIds = new Set(elements.map((e) => e.data.id as string))
    cy.elements().filter((ele) => !targetIds.has(ele.id())).remove()

    const currentIds = new Set(cy.elements().map((ele) => ele.id()))
    const toAdd = elements.filter((e) => !currentIds.has(e.data.id as string))
    if (toAdd.length > 0) {
      cy.add(toAdd)
      cy.layout(buildFcoseLayout(settingsRef.current.nodeSpacing, settingsRef.current.animateLayout)).run()
    }

    // Re-apply selection highlight after expand/collapse
    const sel = selectedNodeRef.current
    if (sel !== null) {
      const node = cy.$(`#cluster-${sel}`)
      if (node.length > 0) {
        cy.elements().unselect()
        node.select()
        applyHighlight(node)
      } else {
        applyChildrenHighlight(sel)
      }
    }

    // Re-apply focus after element changes (no animation — elements already positioned)
    const fs = focusStackRef.current
    if (fs.length > 0) applyFocus(fs[fs.length - 1].id, false)
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
          {(canExpand || canCollapse) && <div className="node-context-separator" />}
          <button className="node-context-item" onClick={() => {
            const { clusterId, level, label } = contextMenu
            const alreadyExpanded = expandedNodes.has(clusterId)
            setFocusStack((s) => [...s, { id: clusterId, label, level }])
            if (level > 0 && !alreadyExpanded) {
              onNodeDblClick(clusterId, level)
              focusAutoExpandedRef.current.add(clusterId)
            }
            setContextMenu(null)
          }}>
            <svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5">
              <circle cx="10" cy="10" r="7"/>
              <line x1="10" y1="1" x2="10" y2="5"/>
              <line x1="10" y1="15" x2="10" y2="19"/>
              <line x1="1" y1="10" x2="5" y2="10"/>
              <line x1="15" y1="10" x2="19" y2="10"/>
            </svg>
            Focus on this cluster
          </button>
          {focusStack.length > 0 && (
            <button className="node-context-item" onClick={() => {
              clearAllFocus()
              setContextMenu(null)
            }}>
              <svg width="14" height="14" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5"><line x1="4" y1="4" x2="16" y2="16"/><line x1="16" y1="4" x2="4" y2="16"/></svg>
              Clear all focus
            </button>
          )}
        </div>
      )}

      {focusStack.length > 0 && (
        <div className="focus-banner">
          <svg width="13" height="13" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.8" style={{ flexShrink: 0 }}>
            <circle cx="10" cy="10" r="7"/>
            <line x1="10" y1="1" x2="10" y2="5"/>
            <line x1="10" y1="15" x2="10" y2="19"/>
            <line x1="1" y1="10" x2="5" y2="10"/>
            <line x1="15" y1="10" x2="19" y2="10"/>
          </svg>
          <span className="focus-banner-label">
            {focusStack[focusStack.length - 1].label}
            {focusStack.length > 1 && <span className="focus-banner-depth"> ({focusStack.length} levels)</span>}
          </span>
          <button
            className="focus-banner-exit"
            title="Exit this focus level"
            onClick={exitFocus}
          >
            ✕
          </button>
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

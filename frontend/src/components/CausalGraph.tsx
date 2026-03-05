import cytoscape, { type ElementDefinition, type StylesheetStyle } from 'cytoscape'
// @ts-expect-error – no types for fcose
import fcose from 'cytoscape-fcose'
import { useCallback, useEffect, useRef, useState } from 'react'
import type { ClusterNode, GraphEdge, SelectedEdge } from '../types'

cytoscape.use(fcose)

const FCOSE_LAYOUT: cytoscape.LayoutOptions = {
  name: 'fcose',
  animate: true,
  animationDuration: 600,
  // Spacing — push nodes well apart so labels don't overlap
  nodeRepulsion: () => 300000,
  idealEdgeLength: () => 350,
  nodeSeparation: 250,
  gravity: 0.04,
  gravityRange: 1.5,
  padding: 60,
  numIter: 5000,
} as cytoscape.LayoutOptions

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

const CYTOSCAPE_STYLE: StylesheetStyle[] = [
  {
    selector: 'node',
    style: {
      label: 'data(label)',
      // Labels sit below the node circle for clarity
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
      // White pill behind text for contrast against any background
      'text-background-color': '#fff',
      'text-background-opacity': 0.88,
      'text-background-shape': 'roundrectangle',
      'text-background-padding': '2px',
      'border-width': 0,
    },
  },
  {
    selector: 'node.level-2',
    style: {
      'background-color': LEVEL_COLORS[2],
      width: 52,
      height: 52,
      'font-size': '12px',
    },
  },
  {
    selector: 'node.level-1',
    style: {
      'background-color': LEVEL_COLORS[1],
      width: 38,
      height: 38,
    },
  },
  {
    selector: 'node.level-0',
    style: {
      'background-color': LEVEL_COLORS[0],
      width: 26,
      height: 26,
      'font-size': '10px',
    },
  },
  {
    // Expanded parent container
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
      width: 'mapData(post_count, 1, 500, 1, 10)',
      'line-color': '#aaaaaa',
      'target-arrow-color': '#aaaaaa',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      opacity: 1,
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
    // Outgoing edges (this node → other): Webis primary blue
    selector: 'edge.edge-outgoing',
    style: {
      'line-color': '#1e87f0',
      'target-arrow-color': '#1e87f0',
      opacity: 1,
      width: 'mapData(post_count, 1, 500, 2, 14)',
    },
  },
  {
    // Incoming edges (other → this node): Webis warning orange
    selector: 'edge.edge-incoming',
    style: {
      'line-color': '#faa05a',
      'target-arrow-color': '#faa05a',
      opacity: 1,
      width: 'mapData(post_count, 1, 500, 2, 14)',
    },
  },
  {
    // Everything else fades when a node is selected
    selector: '.dimmed',
    style: {
      opacity: 0.15,
    },
  },
]

interface CausalGraphProps {
  nodes: ClusterNode[]
  edges: GraphEdge[]
  expandedNodes: Set<number>
  childNodesByParent: Map<number, ClusterNode[]>
  childEdgesByParent: Map<number, GraphEdge[]>
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
  onNodeDblClick,
  onNodeRightClick,
  onNodeClick,
  onEdgeClick,
}: CausalGraphProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const cyRef = useRef<cytoscape.Core | null>(null)
  const [contextMenu, setContextMenu] = useState<{
    x: number; y: number; clusterId: number; level: number
  } | null>(null)

  // Initialize Cytoscape once
  useEffect(() => {
    if (!containerRef.current) return

    cyRef.current = cytoscape({
      container: containerRef.current,
      style: CYTOSCAPE_STYLE,
      layout: FCOSE_LAYOUT,
      userZoomingEnabled: true,
      userPanningEnabled: true,
      boxSelectionEnabled: false,
    })

    return () => {
      cyRef.current?.destroy()
      cyRef.current = null
    }
  }, [])

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
      const { clientX, clientY } = evt.originalEvent as MouseEvent
      setContextMenu({ x: clientX, y: clientY, clusterId: id, level })
    })

    cy.on('tap', 'node', (evt) => {
      const node = evt.target
      const id = parseInt(node.id().replace('cluster-', ''), 10)

      cy.elements().removeClass('edge-outgoing edge-incoming dimmed')
      const outgoing = node.outgoers('edge')
      const incoming = node.incomers('edge')
      const neighborNodes = node.neighborhood('node')
      const kept = outgoing.union(incoming).union(neighborNodes).union(node)
      cy.elements().not(kept).addClass('dimmed')
      outgoing.addClass('edge-outgoing')
      incoming.addClass('edge-incoming')

      onNodeClick(id)
    })

    cy.on('tap', 'edge', (evt) => {
      cy.elements().removeClass('edge-outgoing edge-incoming dimmed')
      const parts = evt.target.id().split('-')
      // edge-{src}-{tgt}
      const src = parseInt(parts[1], 10)
      const tgt = parseInt(parts[2], 10)
      onEdgeClick({ source_cluster_id: src, target_cluster_id: tgt })
    })

    cy.on('tap', (evt) => {
      // Click on background — clear all highlights
      if (evt.target === cy) {
        cy.elements().removeClass('edge-outgoing edge-incoming dimmed')
      }
    })
  }, [onNodeDblClick, onNodeRightClick, onNodeClick, onEdgeClick])

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

    cy.style(CYTOSCAPE_STYLE)
    cy.elements().removeClass('edge-outgoing edge-incoming dimmed').remove()
    cy.add(elements)
    cy.layout(FCOSE_LAYOUT).run()
  }, [nodes, edges, expandedNodes, childNodesByParent, childEdgesByParent])

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
    </div>
  )
}

import cytoscape, { type ElementDefinition, type StylesheetStyle } from 'cytoscape'
// @ts-expect-error – no types for fcose
import fcose from 'cytoscape-fcose'
import { useCallback, useEffect, useRef } from 'react'
import type { ClusterNode, GraphEdge, SelectedEdge } from '../types'

cytoscape.use(fcose)

const CYTOSCAPE_STYLE: StylesheetStyle[] = [
  {
    selector: 'node',
    style: {
      label: 'data(label)',
      'text-valign': 'center',
      'text-halign': 'center',
      'text-wrap': 'wrap',
      'text-max-width': '100px',
      'font-size': '10px',
      color: '#f1f5f9',
      'text-outline-color': '#0f172a',
      'text-outline-width': 2,
      'border-width': 0,
    },
  },
  {
    selector: 'node.level-2',
    style: {
      'background-color': '#2563eb',
      width: 80,
      height: 80,
    },
  },
  {
    selector: 'node.level-1',
    style: {
      'background-color': '#7c3aed',
      width: 55,
      height: 55,
    },
  },
  {
    selector: 'node.level-0',
    style: {
      'background-color': '#059669',
      width: 36,
      height: 36,
    },
  },
  {
    selector: ':compound',
    style: {
      'background-opacity': 0.1,
      'background-color': '#64748b',
      'border-width': 2,
      'border-color': '#64748b',
      padding: '20px',
    },
  },
  {
    selector: 'node:selected',
    style: {
      'border-width': 3,
      'border-color': '#f59e0b',
    },
  },
  {
    selector: 'edge',
    style: {
      width: 'mapData(post_count, 1, 200, 1, 10)',
      'line-color': '#475569',
      'target-arrow-color': '#475569',
      'target-arrow-shape': 'triangle',
      'curve-style': 'bezier',
      opacity: 0.7,
    },
  },
  {
    selector: 'edge:selected',
    style: {
      'line-color': '#f59e0b',
      'target-arrow-color': '#f59e0b',
      opacity: 1,
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

  // Initialize Cytoscape once
  useEffect(() => {
    if (!containerRef.current) return

    cyRef.current = cytoscape({
      container: containerRef.current,
      style: CYTOSCAPE_STYLE,
      layout: { name: 'fcose' },
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
      const id = parseInt(evt.target.id().replace('cluster-', ''), 10)
      onNodeRightClick(id)
    })

    cy.on('tap', 'node', (evt) => {
      const id = parseInt(evt.target.id().replace('cluster-', ''), 10)
      onNodeClick(id)
    })

    cy.on('tap', 'edge', (evt) => {
      const parts = evt.target.id().split('-')
      // edge-{src}-{tgt}
      const src = parseInt(parts[1], 10)
      const tgt = parseInt(parts[2], 10)
      onEdgeClick({ source_cluster_id: src, target_cluster_id: tgt })
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

    cy.elements().remove()
    cy.add(elements)
    cy.layout({ name: 'fcose', animate: true, animationDuration: 400 } as cytoscape.LayoutOptions).run()
  }, [nodes, edges, expandedNodes, childNodesByParent, childEdgesByParent])

  return (
    <div
      ref={containerRef}
      style={{ width: '100%', height: '100%' }}
      title="Double-click to expand · Right-click to collapse · Click edge to see posts"
    />
  )
}

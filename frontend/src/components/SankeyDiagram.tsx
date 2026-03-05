import type { PathLink, PathNode } from '../types'

const NODE_W = 140
const SVG_W = 860
const SVG_H = 360
const PAD_V = 30   // top/bottom padding
const NODE_GAP = 14
const COL_X: Record<string, number> = {
  source: 10,
  intermediate: (SVG_W - NODE_W) / 2,
  target: SVG_W - NODE_W - 10,
}
const LEVEL_COLOR: Record<number, string> = { 2: '#929292', 1: '#b8b8b8', 0: '#d4d4d4' }

interface LayoutNode extends PathNode { x: number; y: number; h: number }

function buildLayout(nodes: PathNode[], links: PathLink[]) {
  // Total flow per node (max of in-flow and out-flow to avoid double-counting)
  const inFlow = new Map<string, number>()
  const outFlow = new Map<string, number>()
  for (const n of nodes) { inFlow.set(n.id, 0); outFlow.set(n.id, 0) }
  for (const l of links) {
    outFlow.set(l.source, (outFlow.get(l.source) ?? 0) + l.post_count)
    inFlow.set(l.target, (inFlow.get(l.target) ?? 0) + l.post_count)
  }
  const flow = (id: string) => Math.max(inFlow.get(id) ?? 1, outFlow.get(id) ?? 1)

  // Group nodes into columns
  const cols: Record<string, PathNode[]> = { source: [], intermediate: [], target: [] }
  for (const n of nodes) cols[n.type === 'source' ? 'source' : n.type === 'target' ? 'target' : 'intermediate'].push(n)

  const layoutNodes: LayoutNode[] = []
  for (const [colName, colNodes] of Object.entries(cols)) {
    if (!colNodes.length) continue
    const x = COL_X[colName]
    const totalFlow = colNodes.reduce((s, n) => s + flow(n.id), 0)
    const available = SVG_H - PAD_V * 2 - NODE_GAP * (colNodes.length - 1)
    let y = PAD_V
    for (const n of colNodes) {
      const h = Math.max(32, (flow(n.id) / totalFlow) * available)
      layoutNodes.push({ ...n, x, y, h })
      y += h + NODE_GAP
    }
  }

  // Stacked ribbon offsets on each node's edge
  const srcOff = new Map(layoutNodes.map(n => [n.id, n.y]))
  const tgtOff = new Map(layoutNodes.map(n => [n.id, n.y]))
  const nodeMap = new Map(layoutNodes.map(n => [n.id, n]))

  const layoutLinks = links.map(l => {
    const src = nodeMap.get(l.source)!
    const tgt = nodeMap.get(l.target)!
    const hSrc = (l.post_count / (outFlow.get(src.id) ?? 1)) * src.h
    const hTgt = (l.post_count / (inFlow.get(tgt.id) ?? 1)) * tgt.h
    const h = Math.max(2, (hSrc + hTgt) / 2)
    const sy = srcOff.get(src.id)!
    const ty = tgtOff.get(tgt.id)!
    srcOff.set(src.id, sy + hSrc)
    tgtOff.set(tgt.id, ty + hTgt)
    return { src, tgt, sy, ty, h, post_count: l.post_count }
  })

  return { layoutNodes, layoutLinks }
}

function ribbon(x1: number, sy: number, x2: number, ty: number, h: number) {
  const mx = (x1 + x2) / 2
  return [
    `M ${x1} ${sy}`,
    `C ${mx} ${sy} ${mx} ${ty} ${x2} ${ty}`,
    `L ${x2} ${ty + h}`,
    `C ${mx} ${ty + h} ${mx} ${sy + h} ${x1} ${sy + h}`,
    'Z',
  ].join(' ')
}

function textLines(s: string, maxLen = 17): string[] {
  const words = s.split(' ')
  const lines: string[] = []
  let cur = ''
  for (const w of words) {
    const next = cur ? `${cur} ${w}` : w
    if (next.length > maxLen && cur) { lines.push(cur); cur = w }
    else cur = next
  }
  if (cur) lines.push(cur)
  return lines
}

export function SankeyDiagram({ nodes, links }: { nodes: PathNode[]; links: PathLink[] }) {
  if (!nodes.length) return null
  const { layoutNodes, layoutLinks } = buildLayout(nodes, links)

  return (
    <div className="sankey-wrap">
      <svg
        viewBox={`0 0 ${SVG_W} ${SVG_H}`}
        className="sankey-svg"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Ribbons */}
        {layoutLinks.map((l, i) => (
          <path
            key={i}
            d={ribbon(l.src.x + NODE_W, l.sy, l.tgt.x, l.ty, l.h)}
            fill="#aaa"
            fillOpacity={0.3}
            stroke="none"
          >
            <title>{l.src.label} → {l.tgt.label}: {l.post_count.toLocaleString()} posts</title>
          </path>
        ))}

        {/* Ribbon post-count labels */}
        {layoutLinks.map((l, i) => {
          const mx = (l.src.x + NODE_W + l.tgt.x) / 2
          const my = (l.sy + l.ty) / 2 + l.h / 2
          return (
            <text key={i} x={mx} y={my} textAnchor="middle" fontSize={10} fill="#777"
              style={{ fontFamily: '"Noto Sans", Verdana, sans-serif' }}>
              {l.post_count.toLocaleString()}
            </text>
          )
        })}

        {/* Node rectangles + labels */}
        {layoutNodes.map(n => {
          const lines = textLines(n.label)
          const lineH = 13
          const totalTextH = lines.length * lineH
          const textY = n.y + Math.max(lineH, (n.h - totalTextH) / 2 + lineH)
          return (
            <g key={n.id}>
              <rect
                x={n.x} y={n.y} width={NODE_W} height={n.h}
                fill={LEVEL_COLOR[n.level] ?? '#ccc'}
                rx={3}
              />
              {lines.map((line, i) => (
                <text
                  key={i}
                  x={n.x + NODE_W / 2}
                  y={textY + i * lineH}
                  textAnchor="middle"
                  fontSize={11}
                  fontWeight={600}
                  fill="#222"
                  style={{ fontFamily: '"Noto Sans", Verdana, sans-serif' }}
                >
                  {line}
                </text>
              ))}
            </g>
          )
        })}

        {/* Column headers */}
        {(['source', 'intermediate', 'target'] as const).map(col => {
          const hasNodes = nodes.some(n =>
            (col === 'intermediate' ? (n.type !== 'source' && n.type !== 'target') : n.type === col)
          )
          if (!hasNodes) return null
          const x = COL_X[col] + NODE_W / 2
          const label = col === 'source' ? 'Cause' : col === 'target' ? 'Effect' : 'Via'
          return (
            <text key={col} x={x} y={16} textAnchor="middle" fontSize={10} fill="#999"
              style={{ fontFamily: '"Noto Sans", Verdana, sans-serif', textTransform: 'uppercase', letterSpacing: '0.08em' }}>
              {label}
            </text>
          )
        })}
      </svg>
    </div>
  )
}

import type { ClusterNode, GraphEdge } from '../types'

interface StatsModalProps {
  open: boolean
  onClose: () => void
  nodes: ClusterNode[]
  edges: GraphEdge[]
  clusterLabels: Map<number, string>
}

function StatRow({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="stat-row">
      <span className="stat-label">{label}</span>
      <span className="stat-value">
        {typeof value === 'number' ? value.toLocaleString() : value}
        {sub && <span className="stat-sub"> {sub}</span>}
      </span>
    </div>
  )
}

function StatSection({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="settings-section">
      <div className="settings-section-title">{title}</div>
      {children}
    </div>
  )
}

export function StatsModal({ open, onClose, nodes, edges, clusterLabels }: StatsModalProps) {
  if (!open) return null

  // Cluster breakdown
  const byLevel = nodes.reduce<Record<number, number>>((acc, n) => {
    acc[n.level] = (acc[n.level] ?? 0) + 1
    return acc
  }, {})

  // Relation totals
  const totalRelations   = edges.reduce((s, e) => s + e.relation_count, 0)
  const totalCountercausal = edges.reduce((s, e) => s + e.countercausal_count, 0)
  const totalCausal      = totalRelations - totalCountercausal
  const ccRate           = totalRelations > 0
    ? ((totalCountercausal / totalRelations) * 100).toFixed(1)
    : '0.0'

  // Graph density (directed): edges / n*(n-1)
  const n = nodes.length
  const possibleEdges = n > 1 ? n * (n - 1) : 1
  const density = ((edges.length / possibleEdges) * 100).toFixed(1)

  // Avg posts per edge
  const totalPosts = edges.reduce((s, e) => s + e.post_count, 0)
  const avgPosts = edges.length > 0 ? (totalPosts / edges.length).toFixed(1) : '—'

  // Top 5 clusters by degree (in + out edges)
  const degree = new Map<number, number>()
  for (const e of edges) {
    degree.set(e.source_cluster_id, (degree.get(e.source_cluster_id) ?? 0) + 1)
    degree.set(e.target_cluster_id, (degree.get(e.target_cluster_id) ?? 0) + 1)
  }
  const topNodes = [...degree.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5)

  // Most evidence: top edges by post_count
  const topEdges = [...edges]
    .sort((a, b) => b.post_count - a.post_count)
    .slice(0, 3)

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Graph Statistics</h2>
          <button className="drawer-close" onClick={onClose} title="Close">✕</button>
        </div>
        <div className="modal-body">

          <StatSection title="Clusters">
            <StatRow label="Total" value={nodes.length} />
            {Object.entries(byLevel).sort((a, b) => Number(b[0]) - Number(a[0])).map(([lvl, count]) => (
              <StatRow
                key={lvl}
                label={`  Level ${lvl} (${lvl === '2' ? 'top' : lvl === '1' ? 'mid' : 'leaf'})`}
                value={count}
              />
            ))}
          </StatSection>

          <StatSection title="Connections">
            <StatRow label="Edges" value={edges.length} />
            <StatRow label="Graph density" value={`${density}%`} />
            <StatRow label="Avg posts / edge" value={avgPosts} />
          </StatSection>

          <StatSection title="Causal Relationships">
            <StatRow label="Total" value={totalRelations} />
            <StatRow label="→  Causal" value={totalCausal}
              sub={totalRelations > 0 ? `(${(100 - Number(ccRate)).toFixed(1)}%)` : ''} />
            <StatRow label="↛  Countercausal" value={totalCountercausal}
              sub={totalRelations > 0 ? `(${ccRate}%)` : ''} />
          </StatSection>

          {topNodes.length > 0 && (
            <StatSection title="Most Connected Clusters">
              {topNodes.map(([id, deg]) => (
                <div key={id} className="stat-row">
                  <span className="stat-label stat-label--trunc">
                    {clusterLabels.get(id) ?? `Cluster ${id}`}
                  </span>
                  <span className="stat-value">{deg} edges</span>
                </div>
              ))}
            </StatSection>
          )}

          {topEdges.length > 0 && (
            <StatSection title="Strongest Edges (by posts)">
              {topEdges.map((e) => {
                const src = clusterLabels.get(e.source_cluster_id) ?? `#${e.source_cluster_id}`
                const tgt = clusterLabels.get(e.target_cluster_id) ?? `#${e.target_cluster_id}`
                return (
                  <div key={`${e.source_cluster_id}-${e.target_cluster_id}`} className="stat-edge-row">
                    <span className="stat-edge-label">{src} → {tgt}</span>
                    <span className="stat-value">{e.post_count.toLocaleString()} posts</span>
                  </div>
                )
              })}
            </StatSection>
          )}

          <StatSection title="Build">
            <StatRow
              label="Version"
              value={import.meta.env.VITE_BUILD_VERSION || 'dev'}
            />
            <StatRow
              label="Built"
              value={import.meta.env.VITE_BUILD_DATE || 'unknown'}
            />
          </StatSection>

        </div>
      </div>
    </div>
  )
}

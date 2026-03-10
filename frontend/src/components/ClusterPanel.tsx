import { useCluster, useClusterPosts } from '../hooks/useGraph'
import { PostItem } from './PostItem'

interface ClusterPanelProps {
  clusterId: number | null
  clusterLabels?: Map<number, string>
  onClusterClick?: (clusterId: number) => void
  isExpanded?: (id: number) => boolean
  onCollapseRequest?: (id: number) => void
  showHighlightSpans?: boolean
  highlightedPostId?: string | null
  onAnalyzePost?: (text: string) => void
}

const LEVEL_META = [
  { label: 'Leaf', cls: 'wlabel wlabel-success' },
  { label: 'Mid',  cls: 'wlabel wlabel-warning'  },
  { label: 'Top',  cls: 'wlabel'                 },
] as const

export function ClusterPanel({ clusterId, clusterLabels, onClusterClick, isExpanded, onCollapseRequest, highlightedPostId, onAnalyzePost }: ClusterPanelProps) {
  const { data, isLoading } = useCluster(clusterId)
  const { data: postsData, isLoading: postsLoading } = useClusterPosts(clusterId)

  if (clusterId === null) {
    return (
      <div className="cluster-panel">
        <p className="hint">Click a node to see details.</p>
        <p className="hint">Double-click to expand sub-clusters.</p>
      </div>
    )
  }

  if (isLoading || !data) {
    return <div className="loading">Loading cluster…</div>
  }

  const levelMeta = LEVEL_META[data.cluster.level] ?? { label: `L${data.cluster.level}`, cls: 'wlabel wlabel-muted' }

  const parentId = data.cluster.parent_id
  const parentLabel = parentId != null ? (clusterLabels?.get(parentId) ?? `Cluster ${parentId}`) : null
  const expanded = isExpanded?.(data.cluster.id) ?? false

  return (
    <div className="cluster-panel">
      <h2>{data.cluster.label}</h2>
      <p className="cluster-panel-meta">
        <span className={levelMeta.cls}>{levelMeta.label}</span>
        {data.cluster.member_count.toLocaleString()} events
        {parentLabel && onClusterClick && (
          <>
            {' · '}
            <button className="internal-link" onClick={() => onClusterClick(parentId!)}>
              {parentLabel}
            </button>
          </>
        )}
      </p>

      {expanded && onCollapseRequest && (
        <button
          className="cluster-collapse-btn"
          onClick={() => onCollapseRequest(data.cluster.id)}
        >
          <svg width="13" height="13" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5"><polyline points="13 4 7 10 13 16"/></svg>
          Collapse sub-clusters
        </button>
      )}

      {data.top_events.length > 0 && (
        <>
          <div className="cluster-section-label">Top events</div>
          <div className="tag-list">
            {data.top_events.slice(0, 12).map((e) => (
              <span key={e} className="tag">{e}</span>
            ))}
          </div>
        </>
      )}

      {data.children.length > 0 && (
        <>
          <div className="cluster-section-label">
            Sub-clusters ({data.children.length})
          </div>
          {data.children.slice(0, 6).map((c) => (
            <div key={c.id} className="subcluster-item">
              <button className="internal-link" onClick={() => onClusterClick?.(c.id)}>
                {c.label}
              </button>
              {' '}
              <span className="subcluster-count">({c.member_count.toLocaleString()})</span>
            </div>
          ))}
        </>
      )}

      {postsLoading && <div className="loading">Loading posts…</div>}

      {postsData && (
        <>
          <div className="cluster-section-label">{postsData.total.toLocaleString()} posts</div>
          {postsData.posts.map((p) => (
            <PostItem
              key={p.id}
              post={p}
              showDate
              highlighted={p.id === highlightedPostId}
              onClusterClick={onClusterClick}
              onAnalyze={onAnalyzePost}
            />
          ))}
        </>
      )}
    </div>
  )
}

import { useCluster } from '../hooks/useGraph'
import type { PostSummary } from '../types'

interface ClusterPanelProps {
  clusterId: number | null
  onExpandRequest: (clusterId: number, level: number) => void
}

const LEVEL_META = [
  { label: 'Leaf', cls: 'wlabel wlabel-success' },
  { label: 'Mid',  cls: 'wlabel wlabel-warning'  },
  { label: 'Top',  cls: 'wlabel'                 },
] as const

function PostItem({ post }: { post: PostSummary }) {
  const date = new Date(post.created_utc * 1000).toLocaleDateString()
  const href = post.permalink
    ? `https://reddit.com${post.permalink}`
    : `https://reddit.com/r/science/comments/${post.id}`

  return (
    <div className="post-item">
      <p className="post-title">
        {post.title}{' '}[<a className="res-link" href={href} target="_blank" rel="noopener noreferrer">reddit</a>]
      </p>
      <span className="post-meta">↑ {post.score} · {post.num_comments} comments · {date}</span>
    </div>
  )
}

export function ClusterPanel({ clusterId, onExpandRequest }: ClusterPanelProps) {
  const { data, isLoading } = useCluster(clusterId)

  if (clusterId === null) {
    return (
      <div className="cluster-panel">
        <p className="hint">Click a node to see details.</p>
        <p className="hint">Double-click to expand / right-click to collapse.</p>
      </div>
    )
  }

  if (isLoading || !data) {
    return <div className="loading">Loading cluster…</div>
  }

  const levelMeta = LEVEL_META[data.cluster.level] ?? { label: `L${data.cluster.level}`, cls: 'wlabel wlabel-muted' }

  return (
    <div className="cluster-panel">
      <h2>{data.cluster.label}</h2>
      <p className="cluster-panel-meta">
        <span className={levelMeta.cls}>{levelMeta.label}</span>
        {data.cluster.member_count.toLocaleString()} events
      </p>

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
            <div
              key={c.id}
              className="subcluster-item"
              onClick={() => onExpandRequest(c.id, c.level)}
            >
              <span className="subcluster-chevron" aria-hidden="true">
                <svg width="16" height="16" viewBox="0 0 20 20" fill="none" stroke="currentColor" strokeWidth="1.5"><polyline points="7 4 13 10 7 16" /></svg>
              </span>
              <span className="subcluster-title">{c.label}</span>
              <span className="subcluster-count">({c.member_count.toLocaleString()})</span>
            </div>
          ))}
        </>
      )}

      {data.posts.length > 0 && (
        <>
          <div className="cluster-section-label">Sample posts</div>
          {data.posts.map((p) => <PostItem key={p.id} post={p} />)}
        </>
      )}
    </div>
  )
}

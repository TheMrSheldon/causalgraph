import { useCluster } from '../hooks/useGraph'
import type { PostSummary } from '../types'

interface ClusterPanelProps {
  clusterId: number | null
  onExpandRequest: (clusterId: number, level: number) => void
}

function PostItem({ post }: { post: PostSummary }) {
  const date = new Date(post.created_utc * 1000).toLocaleDateString()
  const href = post.permalink
    ? `https://reddit.com${post.permalink}`
    : `https://reddit.com/r/science/comments/${post.id}`

  return (
    <div className="post-item">
      <p className="post-title">
        <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: '#93c5fd', textDecoration: 'none' }}>
          {post.title}
        </a>
      </p>
      <span className="post-meta">
        ↑ {post.score} · {post.num_comments} comments · {date}
      </span>
    </div>
  )
}

export function ClusterPanel({ clusterId, onExpandRequest }: ClusterPanelProps) {
  const { data, isLoading } = useCluster(clusterId)

  if (clusterId === null) {
    return (
      <div className="cluster-panel">
        <p className="hint">Click a node to see details.</p>
        <p className="hint">Double-click to expand. Right-click to collapse.</p>
      </div>
    )
  }

  if (isLoading || !data) {
    return <div className="loading">Loading cluster…</div>
  }

  return (
    <div className="cluster-panel">
      <h2>{data.cluster.label}</h2>
      <p className="post-meta" style={{ marginBottom: 8 }}>
        Level {data.cluster.level} · {data.cluster.member_count} events
      </p>

      {data.top_events.length > 0 && (
        <>
          <div style={{ fontSize: 11, color: '#94a3b8', marginBottom: 4 }}>Top events:</div>
          <div className="tag-list">
            {data.top_events.slice(0, 12).map((e) => (
              <span key={e} className="tag">{e}</span>
            ))}
          </div>
        </>
      )}

      {data.children.length > 0 && (
        <>
          <div style={{ fontSize: 11, color: '#94a3b8', margin: '8px 0 4px' }}>
            Sub-clusters ({data.children.length}):
          </div>
          {data.children.slice(0, 6).map((c) => (
            <div
              key={c.id}
              onClick={() => onExpandRequest(c.id, c.level)}
              style={{
                cursor: 'pointer',
                fontSize: 12,
                color: '#a78bfa',
                padding: '2px 0',
              }}
            >
              → {c.label} ({c.member_count})
            </div>
          ))}
        </>
      )}

      {data.posts.length > 0 && (
        <>
          <div style={{ fontSize: 11, color: '#94a3b8', margin: '12px 0 4px' }}>Sample posts:</div>
          {data.posts.map((p) => <PostItem key={p.id} post={p} />)}
        </>
      )}
    </div>
  )
}

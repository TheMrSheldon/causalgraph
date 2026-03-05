import { usePostsForEdge } from '../hooks/usePostsForEdge'
import type { SelectedEdge } from '../types'

interface PostListProps {
  edge: SelectedEdge | null
  onClose: () => void
  sourceLabel?: string
  targetLabel?: string
  onClusterClick: (clusterId: number) => void
}

export function PostList({ edge, onClose, sourceLabel, targetLabel, onClusterClick }: PostListProps) {
  const { data, isLoading } = usePostsForEdge(edge)

  if (!edge) return null

  return (
    <div className="edge-panel">
      <div className="edge-panel-header">
        <h2>Edge posts</h2>
        <button className="drawer-close" onClick={onClose} title="Close">✕</button>
      </div>
      <p className="cluster-panel-meta">
        <button className="internal-link" onClick={() => { onClose(); onClusterClick(edge.source_cluster_id) }}>
          {sourceLabel ?? `Cluster ${edge.source_cluster_id}`}
        </button>
        {' → '}
        <button className="internal-link" onClick={() => { onClose(); onClusterClick(edge.target_cluster_id) }}>
          {targetLabel ?? `Cluster ${edge.target_cluster_id}`}
        </button>
      </p>

      {isLoading && <div className="loading">Loading posts…</div>}

      {data && (
        <>
          <div className="cluster-section-label">{data.total.toLocaleString()} posts</div>
          {data.posts.map((post) => {
            const href = post.permalink
              ? `https://reddit.com${post.permalink}`
              : `https://reddit.com/r/science/comments/${post.id}`
            return (
              <div key={post.id} className="post-item">
                <p className="post-title">
                  {post.title}{' '}[<a className="res-link" href={href} target="_blank" rel="noopener noreferrer">reddit</a>]
                </p>
                <span className="post-meta">↑ {post.score} · {post.num_comments} comments</span>
              </div>
            )
          })}
        </>
      )}
    </div>
  )
}

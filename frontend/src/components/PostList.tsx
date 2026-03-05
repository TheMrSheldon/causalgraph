import { usePostsForEdge } from '../hooks/usePostsForEdge'
import type { SelectedEdge } from '../types'

interface PostListProps {
  edge: SelectedEdge | null
  onClose: () => void
}

export function PostList({ edge, onClose }: PostListProps) {
  const { data, isLoading } = usePostsForEdge(edge)

  if (!edge) return null

  return (
    <div className="post-drawer">
      <div className="post-drawer-header">
        <h3>Posts: cluster {edge.source_cluster_id} → cluster {edge.target_cluster_id}</h3>
        <button className="drawer-close" onClick={onClose} title="Close">✕</button>
      </div>

      {isLoading && <div className="loading">Loading posts…</div>}

      {data && (
        <>
          <div className="post-drawer-count">{data.total.toLocaleString()} posts found</div>
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

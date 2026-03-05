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
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
        <h3>
          Posts: cluster {edge.source_cluster_id} → cluster {edge.target_cluster_id}
        </h3>
        <button
          onClick={onClose}
          style={{
            background: 'none',
            border: 'none',
            color: '#94a3b8',
            cursor: 'pointer',
            fontSize: 18,
          }}
        >
          ✕
        </button>
      </div>

      {isLoading && <div className="loading">Loading posts…</div>}

      {data && (
        <>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 8 }}>
            {data.total} posts found
          </div>
          {data.posts.map((post) => {
            const href = post.permalink
              ? `https://reddit.com${post.permalink}`
              : `https://reddit.com/r/science/comments/${post.id}`
            return (
              <div key={post.id} className="post-item">
                <p className="post-title">
                  <a href={href} target="_blank" rel="noopener noreferrer" style={{ color: '#93c5fd', textDecoration: 'none' }}>
                    {post.title}
                  </a>
                </p>
                <span className="post-meta">
                  ↑ {post.score} · {post.num_comments} comments
                </span>
              </div>
            )
          })}
        </>
      )}
    </div>
  )
}

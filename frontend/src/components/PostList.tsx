import { usePostsForEdge } from '../hooks/usePostsForEdge'
import type { SelectedEdge } from '../types'
import { PostItem } from './PostItem'

interface PostListProps {
  edge: SelectedEdge | null
  sourceLabel?: string
  targetLabel?: string
  onClusterClick: (clusterId: number) => void
  showHighlightSpans: boolean
  highlightedPostId?: string | null
}

export function PostList({ edge, sourceLabel, targetLabel, onClusterClick, showHighlightSpans, highlightedPostId }: PostListProps) {
  const { data, isLoading } = usePostsForEdge(edge)

  if (!edge) return null

  return (
    <div className="edge-panel">
      <div className="edge-panel-header">
        <h2>Edge posts</h2>
      </div>
      <p className="cluster-panel-meta">
        <button className="internal-link" onClick={() => onClusterClick(edge.source_cluster_id)}>
          {sourceLabel ?? `Cluster ${edge.source_cluster_id}`}
        </button>
        {' → '}
        <button className="internal-link" onClick={() => onClusterClick(edge.target_cluster_id)}>
          {targetLabel ?? `Cluster ${edge.target_cluster_id}`}
        </button>
      </p>

      {isLoading && <div className="loading">Loading posts…</div>}

      {data && (
        <>
          <div className="cluster-section-label">{data.total.toLocaleString()} posts</div>
          {data.posts.map((post) => (
            <PostItem key={post.id} post={post} showSpans={showHighlightSpans} highlighted={post.id === highlightedPostId} />
          ))}
        </>
      )}
    </div>
  )
}

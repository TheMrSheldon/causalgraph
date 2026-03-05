import { type ReactNode, useState } from 'react'
import { usePostsForEdge } from '../hooks/usePostsForEdge'
import type { EdgePostSummary, SelectedEdge } from '../types'

interface PostListProps {
  edge: SelectedEdge | null
  onClose: () => void
  sourceLabel?: string
  targetLabel?: string
  onClusterClick: (clusterId: number) => void
}

function highlightSpans(
  title: string,
  causeText: string | null | undefined,
  effectText: string | null | undefined,
): ReactNode {
  type Span = { start: number; end: number; type: 'cause' | 'effect' }
  const spans: Span[] = []

  const findSpan = (text: string | null | undefined, type: 'cause' | 'effect') => {
    if (!text) return
    const idx = title.toLowerCase().indexOf(text.toLowerCase())
    if (idx !== -1) spans.push({ start: idx, end: idx + text.length, type })
  }

  findSpan(causeText, 'cause')
  findSpan(effectText, 'effect')
  if (spans.length === 0) return title

  spans.sort((a, b) => a.start - b.start)

  const parts: ReactNode[] = []
  let pos = 0
  for (const span of spans) {
    if (span.start < pos) continue // overlapping: skip
    if (span.start > pos) parts.push(title.slice(pos, span.start))
    parts.push(
      <mark key={span.start} className={`span-${span.type}`}>
        {title.slice(span.start, span.end)}
      </mark>
    )
    pos = span.end
  }
  if (pos < title.length) parts.push(title.slice(pos))
  return <>{parts}</>
}

function EdgePostItem({ post, showSpans }: { post: EdgePostSummary; showSpans: boolean }) {
  const href = post.permalink
    ? `https://reddit.com${post.permalink}`
    : `https://reddit.com/r/science/comments/${post.id}`

  return (
    <div className="post-item">
      <p className="post-title">
        {showSpans
          ? highlightSpans(post.title, post.cause_text, post.effect_text)
          : post.title}
        {' '}[<a className="res-link" href={href} target="_blank" rel="noopener noreferrer">reddit</a>]
      </p>
      <span className="post-meta">↑ {post.score} · {post.num_comments} comments</span>
    </div>
  )
}

export function PostList({ edge, onClose, sourceLabel, targetLabel, onClusterClick }: PostListProps) {
  const { data, isLoading } = usePostsForEdge(edge)
  const [showSpans, setShowSpans] = useState(true)

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

      <label className="toggle-row span-toggle-row">
        <span className="toggle-label">Highlight event spans</span>
        <button
          className={`toggle-switch${showSpans ? ' toggle-on' : ''}`}
          onClick={() => setShowSpans((v) => !v)}
          role="switch"
          aria-checked={showSpans}
        >
          <span className="toggle-thumb" />
        </button>
      </label>

      {isLoading && <div className="loading">Loading posts…</div>}

      {data && (
        <>
          <div className="cluster-section-label">{data.total.toLocaleString()} posts</div>
          {data.posts.map((post) => (
            <EdgePostItem key={post.id} post={post} showSpans={showSpans} />
          ))}
        </>
      )}
    </div>
  )
}

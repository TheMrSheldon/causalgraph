import { type ReactNode } from 'react'
import type { EdgePostSummary } from '../types'

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
    if (span.start < pos) continue
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

export function PostItem({
  post,
  showSpans = false,
  showDate = false,
}: {
  post: EdgePostSummary
  showSpans?: boolean
  showDate?: boolean
}) {
  const href = post.permalink
    ? `https://reddit.com${post.permalink}`
    : `https://reddit.com/r/science/comments/${post.id}`
  const date = showDate ? new Date(post.created_utc * 1000).toLocaleDateString() : null

  return (
    <div className="post-item">
      <p className="post-title">
        {showSpans
          ? highlightSpans(post.title, post.cause_text, post.effect_text)
          : post.title}
        {' '}[<a className="res-link" href={href} target="_blank" rel="noopener noreferrer">reddit</a>]
        {post.is_countercausal && <span className="countercausal-badge">refuted</span>}
      </p>
      <span className="post-meta">
        ↑ {post.score} · {post.num_comments} comments{date ? ` · ${date}` : ''}
      </span>
    </div>
  )
}

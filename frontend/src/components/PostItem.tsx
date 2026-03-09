import { createPortal } from 'react-dom'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { buildShareUrl } from '../hooks/useUrlSync'
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

// ── Share popup (rendered in a portal so sidebar overflow doesn't clip it) ──

function SharePopup({
  postId,
  anchorRect,
  onClose,
}: {
  postId: string
  anchorRect: DOMRect
  onClose: () => void
}) {
  const [includeState, setIncludeState] = useState(false)
  const [copied, setCopied] = useState(false)
  const popupRef = useRef<HTMLDivElement>(null)

  const url = useMemo(
    () => buildShareUrl(postId, includeState),
    [postId, includeState]
  )

  // Position: below the anchor button, right-aligned
  const style: React.CSSProperties = {
    position: 'fixed',
    top:   anchorRect.bottom + 6,
    right: window.innerWidth - anchorRect.right,
    zIndex: 9999,
  }

  function copy() {
    navigator.clipboard.writeText(url).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    }).catch(() => {
      // Fallback for insecure contexts
      const el = document.createElement('textarea')
      el.value = url
      document.body.appendChild(el)
      el.select()
      document.execCommand('copy')
      document.body.removeChild(el)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
    })
  }

  // Dismiss on outside mousedown
  useEffect(() => {
    function onMouseDown(e: MouseEvent) {
      if (popupRef.current && !popupRef.current.contains(e.target as Node)) {
        onClose()
      }
    }
    // Slight delay so the click that opened the popup doesn't immediately close it
    const tid = setTimeout(() => document.addEventListener('mousedown', onMouseDown), 0)
    return () => {
      clearTimeout(tid)
      document.removeEventListener('mousedown', onMouseDown)
    }
  }, [onClose])

  return createPortal(
    <div className="share-popup" ref={popupRef} style={style}>
      <div className="share-popup-toggles">
        <button
          className={`share-toggle-btn${!includeState ? ' share-toggle-btn--active' : ''}`}
          onClick={() => setIncludeState(false)}
        >
          Post only
        </button>
        <button
          className={`share-toggle-btn${includeState ? ' share-toggle-btn--active' : ''}`}
          onClick={() => setIncludeState(true)}
        >
          With graph state
        </button>
      </div>
      <div className="share-popup-url-row">
        <input
          className="share-popup-url"
          value={url}
          readOnly
          onClick={(e) => (e.target as HTMLInputElement).select()}
          spellCheck={false}
        />
        <button className="share-copy-btn" onClick={copy}>
          {copied ? '✓' : 'Copy'}
        </button>
      </div>
    </div>,
    document.body
  )
}

// ── PostItem ──────────────────────────────────────────────────────────────────

export function PostItem({
  post,
  showSpans = false,
  showDate = false,
  highlighted = false,
}: {
  post: EdgePostSummary
  showSpans?: boolean
  showDate?: boolean
  /** When true: scroll into view and flash-highlight on mount */
  highlighted?: boolean
}) {
  const href = post.permalink
    ? `https://reddit.com${post.permalink}`
    : `https://reddit.com/r/science/comments/${post.id}`
  const date = showDate ? new Date(post.created_utc * 1000).toLocaleDateString() : null

  const itemRef    = useRef<HTMLDivElement>(null)
  const shareBtnRef = useRef<HTMLButtonElement>(null)
  const [shareRect, setShareRect] = useState<DOMRect | null>(null)

  // Scroll + flash on highlighted mount
  useEffect(() => {
    if (highlighted && itemRef.current) {
      itemRef.current.scrollIntoView({ behavior: 'smooth', block: 'center' })
    }
  }, [highlighted])

  function toggleShare(e: React.MouseEvent) {
    e.stopPropagation()
    if (shareRect) {
      setShareRect(null)
      return
    }
    const rect = shareBtnRef.current?.getBoundingClientRect()
    if (rect) setShareRect(rect)
  }

  return (
    <div
      ref={itemRef}
      className={`post-item${highlighted ? ' post-item--highlighted' : ''}`}
    >
      <p className="post-title">
        {showSpans
          ? highlightSpans(post.title, post.cause_text, post.effect_text)
          : post.title}
        {' '}[<a className="res-link" href={href} target="_blank" rel="noopener noreferrer">reddit</a>]
        {post.is_countercausal && <span className="countercausal-badge">refuted</span>}
      </p>
      <div className="post-meta-row">
        <span className="post-meta">
          ↑ {post.score} · {post.num_comments} comments{date ? ` · ${date}` : ''}
        </span>
        <button
          ref={shareBtnRef}
          className={`share-btn${shareRect ? ' share-btn--active' : ''}`}
          onClick={toggleShare}
          title="Share"
        >
          {/* Chain-link icon */}
          <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/>
            <path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/>
          </svg>
        </button>
      </div>

      {shareRect && (
        <SharePopup
          postId={post.id}
          anchorRect={shareRect}
          onClose={() => setShareRect(null)}
        />
      )}
    </div>
  )
}

import { createPortal } from 'react-dom'
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react'
import { buildShareUrl } from '../hooks/useUrlSync'
import type { EdgePostSummary, RelationSpan } from '../types'

// ---------------------------------------------------------------------------
// Event span rendering (wavy underlines with tooltip + click-to-navigate)
// ---------------------------------------------------------------------------

type EventInfo = {
  start: number
  end: number
  canonical: string
  clusterId: number | null
  role: 'cause' | 'effect'
}

/** Find the first occurrence of `needle` in `haystack` (case-insensitive). */
function findPos(haystack: string, needle: string): number {
  return haystack.toLowerCase().indexOf(needle.toLowerCase())
}

/**
 * Collect all event spans from all relations, deduplicated by (start, end, role).
 * Multiple relations can map to the same text span in the title.
 */
function collectEvents(title: string, relations: RelationSpan[]): EventInfo[] {
  const seen = new Set<string>()
  const events: EventInfo[] = []
  for (const rel of relations) {
    const addSpan = (text: string, role: 'cause' | 'effect', canonical: string | null, clusterId: number | null) => {
      const idx = findPos(title, text)
      if (idx === -1) return
      const key = `${idx}:${idx + text.length}:${role}`
      if (seen.has(key)) return
      seen.add(key)
      events.push({ start: idx, end: idx + text.length, canonical: canonical || text, clusterId, role })
    }
    addSpan(rel.cause_text, 'cause', rel.cause_canonical, rel.cause_cluster_id)
    addSpan(rel.effect_text, 'effect', rel.effect_canonical, rel.effect_cluster_id)
  }
  return events
}

/** Single styled span segment covering [start, end) in the title. */
function EventSpan({
  text,
  events,
  onClusterClick,
}: {
  text: string
  events: EventInfo[]  // all events that contain this segment, longest-first
  onClusterClick?: (clusterId: number) => void
}) {
  const [tipVisible, setTipVisible] = useState(false)
  // Innermost (shortest) event provides the visual style
  const inner = events[events.length - 1]
  const clickable = inner.clusterId != null && onClusterClick != null

  const cls = [
    'event-span',
    `event-span--${inner.role}`,
    clickable ? 'event-span--clickable' : '',
  ].filter(Boolean).join(' ')

  // Tooltip: one line per distinct canonical description
  const tipLines = [...new Set(events.map(e => e.canonical))].join('\n')

  return (
    <span
      className={cls}
      onMouseEnter={() => setTipVisible(true)}
      onMouseLeave={() => setTipVisible(false)}
      onClick={clickable ? (e) => { e.stopPropagation(); onClusterClick!(inner.clusterId!) } : undefined}
    >
      {text}
      {tipVisible && (
        <span className="event-span-tooltip">{tipLines}</span>
      )}
    </span>
  )
}

/**
 * Render title text with event spans as wavy-underlined, hoverable, clickable
 * elements. Handles nested and overlapping spans via boundary segmentation.
 */
function renderEventSpans(
  title: string,
  relations: RelationSpan[],
  onClusterClick?: (clusterId: number) => void,
): ReactNode {
  const events = collectEvents(title, relations)
  if (!events.length) return title

  // Collect all boundary points across all spans
  const pts = Array.from(
    new Set([0, title.length, ...events.flatMap(e => [e.start, e.end])])
  ).sort((a, b) => a - b)

  const parts: ReactNode[] = []
  for (let i = 0; i < pts.length - 1; i++) {
    const s = pts[i], e = pts[i + 1]
    // Events that fully contain this segment
    const active = events.filter(ev => ev.start <= s && ev.end >= e)
    if (!active.length) {
      parts.push(title.slice(s, e))
    } else {
      // Sort outer→inner (longest first); innermost provides visual role
      active.sort((a, b) => (b.end - b.start) - (a.end - a.start))
      parts.push(
        <EventSpan
          key={s}
          text={title.slice(s, e)}
          events={active}
          onClusterClick={onClusterClick}
        />
      )
    }
  }
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
  showDate = false,
  highlighted = false,
  onClusterClick,
  onAnalyze,
}: {
  post: EdgePostSummary
  showDate?: boolean
  /** When true: scroll into view and flash-highlight on mount */
  highlighted?: boolean
  /** Navigate to a cluster when an event span is clicked */
  onClusterClick?: (clusterId: number) => void
  /** Navigate to the Text Analyzer with this post's title pre-filled */
  onAnalyze?: (text: string) => void
}) {
  const href = post.permalink
    ? `https://reddit.com${post.permalink}`
    : `https://reddit.com/r/science/comments/${post.id}`
  const date = showDate ? new Date(post.created_utc * 1000).toLocaleDateString(undefined, { day: '2-digit', month: '2-digit', year: 'numeric' }) : null

  const itemRef    = useRef<HTMLDivElement>(null)
  const shareBtnRef = useRef<HTMLButtonElement>(null)
  const [shareRect, setShareRect] = useState<DOMRect | null>(null)

  const relations = post.relations ?? []
  const isCountercausal = relations.some(r => r.is_countercausal)

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
        {renderEventSpans(post.title, relations, onClusterClick)}
        {' '}[<a className="res-link" href={href} target="_blank" rel="noopener noreferrer">reddit</a>]
        {isCountercausal && <span className="countercausal-badge">refuted</span>}
      </p>
      <div className="post-meta-row">
        <span className="post-meta">
          ↑ {post.score} · {post.num_comments} comments{date ? ` · ${date}` : ''}
        </span>
        <div style={{ display: 'flex', gap: 4, marginLeft: 'auto' }}>
          {onAnalyze && (
            <button
              className="share-btn"
              onClick={(e) => { e.stopPropagation(); onAnalyze(post.title) }}
              title="Analyze"
            >
              {/* Magnifying-glass icon */}
              <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <circle cx="11" cy="11" r="8"/>
                <line x1="21" y1="21" x2="16.65" y2="16.65"/>
              </svg>
            </button>
          )}
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

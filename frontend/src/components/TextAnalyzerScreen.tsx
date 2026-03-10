import { type ReactNode, useState } from 'react'
import { api } from '../api/client'
import type { AnalysisEvent, AnalysisRelation, AnalysisResponse } from '../types'

// ---------------------------------------------------------------------------
// Per-event palette
// ---------------------------------------------------------------------------
const EVENT_PALETTE = [
  '#fde68a', '#a7f3d0', '#bfdbfe', '#fca5a5', '#c4b5fd',
  '#fdba74', '#6ee7b7', '#93c5fd', '#f9a8d4', '#d9f99d',
  '#fef08a', '#bbf7d0', '#ddd6fe', '#fed7aa', '#cffafe',
]

function eventColor(index: number) {
  return EVENT_PALETTE[index % EVENT_PALETTE.length]
}

// ---------------------------------------------------------------------------
// Highlighted text — dims spans not in activeIndices
// ---------------------------------------------------------------------------
function renderHighlighted(
  text: string,
  events: AnalysisEvent[],
  activeIndices: Set<number> | null,
): ReactNode {
  if (!events.length) return text
  const sorted = [...events].sort((a, b) => a.start - b.start)
  const parts: ReactNode[] = []
  let pos = 0
  for (const ev of sorted) {
    if (ev.start < pos) continue
    if (ev.start > pos) parts.push(text.slice(pos, ev.start))
    const isActive = activeIndices === null || activeIndices.has(ev.index)
    parts.push(
      <mark
        key={ev.start}
        style={{
          backgroundColor: isActive ? eventColor(ev.index) : '#e2e8f0',
          opacity: isActive ? 1 : 0.35,
          borderRadius: 2,
          padding: '0 1px',
          transition: 'background-color 0.12s, opacity 0.12s',
        }}
      >
        {text.slice(ev.start, ev.end)}
      </mark>
    )
    pos = ev.end
  }
  if (pos < text.length) parts.push(text.slice(pos))
  return <>{parts}</>
}

// ---------------------------------------------------------------------------
// Events legend — clickable chips with hover/selection state
// ---------------------------------------------------------------------------
function EventsLegend({
  events,
  selectedIndices,
  hoveredIndex,
  onToggle,
  onHover,
}: {
  events: AnalysisEvent[]
  selectedIndices: Set<number>
  hoveredIndex: number | null
  onToggle: (index: number) => void
  onHover: (index: number | null) => void
}) {
  if (!events.length) return null
  const anySelected = selectedIndices.size > 0

  return (
    <div className="analyzer-events-legend">
      <span className="analyzer-events-legend-label">Detected spans</span>
      {events.map(ev => {
        const selected = selectedIndices.has(ev.index)
        const dimmed = anySelected && !selected && hoveredIndex !== ev.index
        return (
          <button
            key={ev.index}
            className={`analyzer-event-chip${selected ? ' analyzer-event-chip--selected' : ''}`}
            style={{ opacity: dimmed ? 0.4 : 1 }}
            onMouseEnter={() => onHover(ev.index)}
            onMouseLeave={() => onHover(null)}
            onClick={() => onToggle(ev.index)}
            title={selected ? 'Click to deselect' : 'Click to filter by this span'}
          >
            <span className="analyzer-event-dot" style={{ background: eventColor(ev.index) }} />
            {ev.description || ev.span_text}
          </button>
        )
      })}
      {anySelected && (
        <button
          className="analyzer-event-chip analyzer-event-chip--clear"
          onClick={() => { events.forEach(ev => selectedIndices.has(ev.index) && onToggle(ev.index)) }}
          title="Clear filter"
        >
          ✕ clear
        </button>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Probability bar
// ---------------------------------------------------------------------------
function ProbBar({ label, value, color }: { label: string; value: number; color: string }) {
  const pct = Math.round(value * 100)
  return (
    <div className="prob-row">
      <span className="prob-label">{label}</span>
      <div className="prob-track">
        <div className="prob-fill" style={{ width: `${pct}%`, background: color }} />
      </div>
      <span className="prob-value">{pct}%</span>
    </div>
  )
}

function ProbTooltip({ rel }: { rel: AnalysisRelation }) {
  return (
    <div className="prob-tooltip">
      <ProbBar label="causal"    value={rel.p_causal}        color="#4ade80" />
      <ProbBar label="refuted"   value={rel.p_countercausal} color="#f87171" />
      <ProbBar label="unrelated" value={rel.p_none}          color="#94a3b8" />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Relation row
// ---------------------------------------------------------------------------
function RelationRow({
  rel,
  events,
  selectedIndices,
}: {
  rel: AnalysisRelation
  events: AnalysisEvent[]
  selectedIndices: Set<number>
}) {
  const [showProbs, setShowProbs] = useState(false)
  const isNoRel = rel.relation_type === 'no_rel'


  const causeLabel = rel.cause_canonical || events[rel.cause_event_index]?.description || rel.cause_text
  const effectLabel = rel.effect_canonical || events[rel.effect_event_index]?.description || rel.effect_text
  const causeRaw = rel.cause_text.toLowerCase() !== causeLabel.toLowerCase() ? rel.cause_text : null
  const effectRaw = rel.effect_text.toLowerCase() !== effectLabel.toLowerCase() ? rel.effect_text : null

  const arrow = isNoRel ? '≁' : rel.relation_type === 'countercausal' ? '↛' : '→'
  const arrowTitle = isNoRel ? 'no relation' : rel.relation_type === 'countercausal' ? 'refuted' : 'causes'

  // Dim endpoints not in the active selection
  const anySelected = selectedIndices.size > 0
  const causeDimmed = anySelected && !selectedIndices.has(rel.cause_event_index)
  const effectDimmed = anySelected && !selectedIndices.has(rel.effect_event_index)

  return (
    <div
      className={`analyzer-relation${isNoRel ? ' analyzer-relation--no-rel' : ''}${showProbs ? ' analyzer-relation--expanded' : ''}`}
      onClick={() => setShowProbs(v => !v)}
      title={showProbs ? 'Click to hide certainty scores' : 'Click to show certainty scores'}
      style={{ cursor: 'pointer' }}
    >
      <span className="analyzer-relation-row">
        <span className="analyzer-event-cell" style={{ opacity: causeDimmed ? 0.45 : 1, transition: 'opacity 0.12s' }}>
          <mark style={{ background: isNoRel ? '#f1f5f9' : eventColor(rel.cause_event_index) }}>
            {causeLabel}
          </mark>
          {causeRaw && <span className="analyzer-span-raw">{causeRaw}</span>}
        </span>
        <span
          className={`analyzer-relation-arrow${isNoRel ? ' analyzer-relation-arrow--no-rel' : ''}`}
          title={arrowTitle}
        >
          {arrow}
        </span>
        <span className="analyzer-event-cell" style={{ opacity: effectDimmed ? 0.45 : 1, transition: 'opacity 0.12s' }}>
          <mark style={{ background: isNoRel ? '#f1f5f9' : eventColor(rel.effect_event_index) }}>
            {effectLabel}
          </mark>
          {effectRaw && <span className="analyzer-span-raw">{effectRaw}</span>}
        </span>
        {rel.relation_type === 'countercausal' && <span className="countercausal-badge">refuted</span>}
        <span className="analyzer-chevron" aria-hidden>›</span>
      </span>
      {showProbs && <ProbTooltip rel={rel} />}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Grouped relations display
// ---------------------------------------------------------------------------
function RelationsDisplay({
  relations,
  events,
  selectedIndices,
}: {
  relations: AnalysisRelation[]
  events: AnalysisEvent[]
  selectedIndices: Set<number>
}) {
  const [noRelOpen, setNoRelOpen] = useState(false)

  const isVisible = (r: AnalysisRelation) => {
    if (selectedIndices.size === 0) return true
    if (selectedIndices.size === 1)
      return selectedIndices.has(r.cause_event_index) || selectedIndices.has(r.effect_event_index)
    return selectedIndices.has(r.cause_event_index) && selectedIndices.has(r.effect_event_index)
  }

  const causal        = relations.filter(r => r.relation_type !== 'no_rel')
  const noRel         = relations.filter(r => r.relation_type === 'no_rel')
  const visibleCausal = causal.filter(isVisible)
  const visibleNoRel  = noRel.filter(isVisible)

  // Auto-expand no-rel section when filter would produce results there
  const effectiveNoRelOpen = noRelOpen || (selectedIndices.size > 0 && visibleNoRel.length > 0)

  if (relations.length === 0) {
    return (
      <p className="hint">
        No causal relationships detected. Try sentences with words like "causes", "leads to", "increases", "reduces".
      </p>
    )
  }

  const filterActive = selectedIndices.size > 0
  const nothingVisible = visibleCausal.length === 0 && visibleNoRel.length === 0

  return (
    <>
      {(visibleCausal.length > 0 || (!filterActive && causal.length > 0)) && (
        <>
          <div className="cluster-section-label" style={{ marginTop: 16 }}>
            Relationships
            {!filterActive && <span className="analyzer-hover-hint">click for certainty</span>}
            {filterActive && visibleCausal.length > 0 && (
              <span className="analyzer-hover-hint">{visibleCausal.length} of {causal.length}</span>
            )}
          </div>
          {(filterActive ? visibleCausal : causal).map((r, i) => (
            <RelationRow key={i} rel={r} events={events} selectedIndices={selectedIndices} />
          ))}
          {filterActive && visibleCausal.length === 0 && (
            <p className="hint" style={{ marginTop: 4 }}>No relationships between the selected spans.</p>
          )}
        </>
      )}

      {(noRel.length > 0) && (
        <>
          <button
            className={`analyzer-no-rel-toggle${effectiveNoRelOpen ? ' analyzer-no-rel-toggle--open' : ''}`}
            onClick={() => setNoRelOpen(v => !v)}
          >
            <span className="analyzer-chevron" aria-hidden>›</span>
            {filterActive
              ? `${visibleNoRel.length} of ${noRel.length} no-rel pair${noRel.length !== 1 ? 's' : ''} match`
              : `${noRel.length} pair${noRel.length !== 1 ? 's' : ''} with no causal link`}
            {!filterActive && <span className="analyzer-hover-hint">click for certainty</span>}
          </button>
          {effectiveNoRelOpen && (filterActive ? visibleNoRel : noRel).map((r, i) => (
            <RelationRow key={i} rel={r} events={events} selectedIndices={selectedIndices} />
          ))}
        </>
      )}

      {filterActive && nothingVisible && (
        <p className="hint" style={{ marginTop: 8 }}>No relationships between the selected spans.</p>
      )}
    </>
  )
}

// ---------------------------------------------------------------------------
// Example text
// ---------------------------------------------------------------------------
const EXAMPLE = `Exercise reduces the risk of heart disease and improves mental health.
Chronic sleep deprivation increases the likelihood of obesity and leads to cognitive decline.
Air pollution causes respiratory problems and triggers inflammation.
Study finds no evidence that moderate alcohol consumption increases cancer risk.
The likelihood that a cat causes reduction in vascular disease effects is in my opinion essentially nil.`

// ---------------------------------------------------------------------------
// Main screen
// ---------------------------------------------------------------------------
export function TextAnalyzerScreen({
  text,
  onTextChange,
}: {
  text: string
  onTextChange: (t: string) => void
}) {
  const [result, setResult] = useState<AnalysisResponse | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [selectedIndices, setSelectedIndices] = useState<Set<number>>(new Set())
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null)

  const handleAnalyze = async () => {
    if (!text.trim()) return
    setIsAnalyzing(true)
    setError(null)
    setSelectedIndices(new Set())
    try {
      const data = await api.analyzeText(text)
      setResult(data)
    } catch {
      setError('Analysis failed. Is the backend running?')
    } finally {
      setIsAnalyzing(false)
    }
  }

  function toggleEvent(index: number) {
    setSelectedIndices(prev => {
      const next = new Set(prev)
      if (next.has(index)) next.delete(index)
      else next.add(index)
      return next
    })
  }

  // What to highlight in the text preview:
  // hover takes precedence; then selection; then show all
  const activeIndices: Set<number> | null =
    hoveredIndex !== null ? new Set([hoveredIndex]) :
    selectedIndices.size > 0 ? selectedIndices :
    null

  const causalCount = result ? result.relations.filter(r => r.relation_type !== 'no_rel').length : 0
  const noRelCount  = result ? result.relations.filter(r => r.relation_type === 'no_rel').length  : 0
  const metaParts: string[] = []
  if (causalCount > 0) metaParts.push(`${causalCount} relationship${causalCount !== 1 ? 's' : ''}`)
  if (noRelCount  > 0) metaParts.push(`${noRelCount} no-rel pair${noRelCount !== 1 ? 's' : ''}`)
  if (result && result.events.length > 0) metaParts.push(`${result.events.length} span${result.events.length !== 1 ? 's' : ''}`)

  return (
    <div className="analyzer-screen">
      <div className="analyzer-pane-header analyzer-pane-header--input">
        <span className="analyzer-pane-title">Input text</span>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            className="analyzer-btn-secondary"
            onClick={() => { onTextChange(EXAMPLE); setResult(null); setError(null); setSelectedIndices(new Set()) }}
          >
            Load example
          </button>
          <button
            className="analyzer-btn-primary"
            onClick={handleAnalyze}
            disabled={!text.trim() || isAnalyzing}
          >
            {isAnalyzing ? 'Analyzing…' : 'Analyze'}
          </button>
        </div>
      </div>

      <div className="analyzer-pane-header analyzer-pane-header--result">
        <span className="analyzer-pane-title">Analysis</span>
        {result && metaParts.length > 0 && (
          <span className="analyzer-pane-meta">
            {metaParts.join(' · ')}
            {selectedIndices.size > 0 && ` · ${selectedIndices.size} span${selectedIndices.size !== 1 ? 's' : ''} selected`}
          </span>
        )}
      </div>

      <div className="analyzer-pane-body analyzer-pane-body--input">
        <textarea
          className="analyzer-textarea"
          value={text}
          onChange={e => { onTextChange(e.target.value); setResult(null); setSelectedIndices(new Set()) }}
          placeholder="Paste or type text containing causal statements…"
          spellCheck={false}
        />
      </div>

      <div className="analyzer-pane-body">

        {isAnalyzing && <div className="loading">Analyzing…</div>}

        {error && <p className="hint" style={{ color: '#b91c1c' }}>{error}</p>}

        {result && !isAnalyzing && (
          <>
            <div className="analyzer-highlighted-text">
              {renderHighlighted(result.text, result.events, activeIndices)}
            </div>

            <EventsLegend
              events={result.events}
              selectedIndices={selectedIndices}
              hoveredIndex={hoveredIndex}
              onToggle={toggleEvent}
              onHover={setHoveredIndex}
            />

            <RelationsDisplay
              relations={result.relations}
              events={result.events}
              selectedIndices={selectedIndices}
            />
          </>
        )}

        {!result && !isAnalyzing && !error && (
          <p className="hint">
            Causal spans will be highlighted and extracted cause→effect relationships will appear here.
          </p>
        )}

      </div>
    </div>
  )
}

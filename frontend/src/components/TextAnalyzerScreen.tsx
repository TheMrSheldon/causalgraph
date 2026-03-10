import { type ReactNode, useState } from 'react'
import { api } from '../api/client'
import type { AnalysisEvent, AnalysisRelation, AnalysisResponse } from '../types'

// ---------------------------------------------------------------------------
// Per-event palette (background colours for text highlighting)
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
// Highlighted text rendering
// ---------------------------------------------------------------------------
function renderHighlighted(text: string, events: AnalysisEvent[]): ReactNode {
  if (!events.length) return text
  const sorted = [...events].sort((a, b) => a.start - b.start)
  const parts: ReactNode[] = []
  let pos = 0
  for (const ev of sorted) {
    if (ev.start < pos) continue
    if (ev.start > pos) parts.push(text.slice(pos, ev.start))
    parts.push(
      <mark key={ev.start} style={{ backgroundColor: eventColor(ev.index), borderRadius: 2, padding: '0 1px' }}>
        {text.slice(ev.start, ev.end)}
      </mark>
    )
    pos = ev.end
  }
  if (pos < text.length) parts.push(text.slice(pos))
  return <>{parts}</>
}

// ---------------------------------------------------------------------------
// Probability bar (inline in hover tooltip)
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
      <ProbBar label="causal" value={rel.p_causal} color="#4ade80" />
      <ProbBar label="refuted" value={rel.p_countercausal} color="#f87171" />
      <ProbBar label="unrelated" value={rel.p_none} color="#94a3b8" />
    </div>
  )
}

// ---------------------------------------------------------------------------
// Relation row with hover probability panel
// ---------------------------------------------------------------------------
function RelationRow({ rel, events }: { rel: AnalysisRelation; events: AnalysisEvent[] }) {
  const [showProbs, setShowProbs] = useState(false)

  // Prefer canonical descriptions; fall back to span description / raw text
  const causeLabel = rel.cause_canonical || events[rel.cause_event_index]?.description || rel.cause_text
  const effectLabel = rel.effect_canonical || events[rel.effect_event_index]?.description || rel.effect_text
  // Show raw span as secondary only when it meaningfully differs from the canonical
  const causeRaw = rel.cause_text.toLowerCase() !== causeLabel.toLowerCase() ? rel.cause_text : null
  const effectRaw = rel.effect_text.toLowerCase() !== effectLabel.toLowerCase() ? rel.effect_text : null

  return (
    <div
      className="analyzer-relation"
      onMouseEnter={() => setShowProbs(true)}
      onMouseLeave={() => setShowProbs(false)}
    >
      <span className="analyzer-relation-row">
        <span className="analyzer-event-cell">
          <mark className="span-cause" style={{ background: eventColor(rel.cause_event_index) }}>
            {causeLabel}
          </mark>
          {causeRaw && <span className="analyzer-span-raw">{causeRaw}</span>}
        </span>
        <span className="analyzer-relation-arrow" title={rel.is_countercausal ? 'refuted' : 'causes'}>
          {rel.is_countercausal ? '↛' : '→'}
        </span>
        <span className="analyzer-event-cell">
          <mark className="span-effect" style={{ background: eventColor(rel.effect_event_index) }}>
            {effectLabel}
          </mark>
          {effectRaw && <span className="analyzer-span-raw">{effectRaw}</span>}
        </span>
        {rel.is_countercausal && <span className="countercausal-badge">refuted</span>}
      </span>
      {showProbs && <ProbTooltip rel={rel} />}
    </div>
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

  const handleAnalyze = async () => {
    if (!text.trim()) return
    setIsAnalyzing(true)
    setError(null)
    try {
      const data = await api.analyzeText(text)
      setResult(data)
    } catch {
      setError('Analysis failed. Is the backend running?')
    } finally {
      setIsAnalyzing(false)
    }
  }

  return (
    <div className="analyzer-screen">
      <div className="analyzer-pane-header analyzer-pane-header--input">
        <span className="analyzer-pane-title">Input text</span>
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            className="analyzer-btn-secondary"
            onClick={() => { onTextChange(EXAMPLE); setResult(null); setError(null) }}
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
        {result && (
          <span className="analyzer-pane-meta">
            {result.relations.length} relationship{result.relations.length !== 1 ? 's' : ''} found
          </span>
        )}
      </div>

      <div className="analyzer-pane-body analyzer-pane-body--input">
        <textarea
          className="analyzer-textarea"
          value={text}
          onChange={e => { onTextChange(e.target.value); setResult(null) }}
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
              {renderHighlighted(result.text, result.events)}
            </div>

            {result.relations.length > 0 ? (
              <>
                <div className="cluster-section-label" style={{ marginTop: 16 }}>
                  Extracted relationships
                  <span className="analyzer-hover-hint">hover for certainty</span>
                </div>
                {result.relations.map((r, i) => (
                  <RelationRow key={i} rel={r} events={result.events} />
                ))}
              </>
            ) : (
              <p className="hint">No causal relationships detected. Try sentences with words like "causes", "leads to", "increases", "reduces".</p>
            )}
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

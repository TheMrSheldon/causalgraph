import { type ReactNode, useState } from 'react'
import { api } from '../api/client'
import type { AnalysisResponse, AnalysisSpan } from '../types'

function renderHighlighted(text: string, spans: AnalysisSpan[]): ReactNode {
  if (!spans.length) return text
  const sorted = [...spans].sort((a, b) => a.start - b.start)
  const parts: ReactNode[] = []
  let pos = 0
  for (const span of sorted) {
    if (span.start < pos) continue
    if (span.start > pos) parts.push(text.slice(pos, span.start))
    parts.push(
      <mark key={span.start} className={`span-${span.type}`}>
        {text.slice(span.start, span.end)}
      </mark>
    )
    pos = span.end
  }
  if (pos < text.length) parts.push(text.slice(pos))
  return <>{parts}</>
}

const EXAMPLE = `Exercise reduces the risk of heart disease and improves mental health.
Chronic sleep deprivation increases the likelihood of obesity and leads to cognitive decline.
Air pollution causes respiratory problems and triggers inflammation.`

export function TextAnalyzerScreen() {
  const [text, setText] = useState('')
  const [result, setResult] = useState<AnalysisResponse | null>(null)
  const [isAnalyzing, setIsAnalyzing] = useState(false)

  const handleAnalyze = async () => {
    if (!text.trim()) return
    setIsAnalyzing(true)
    const data = await api.analyzeText(text)
    setResult(data)
    setIsAnalyzing(false)
  }

  return (
    <div className="analyzer-screen">
      <div className="analyzer-pane analyzer-pane--input">
        <div className="analyzer-pane-header">
          <span className="analyzer-pane-title">Input text</span>
          <div style={{ display: 'flex', gap: 8 }}>
            <button
              className="analyzer-btn-secondary"
              onClick={() => { setText(EXAMPLE); setResult(null) }}
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
        <textarea
          className="analyzer-textarea"
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Paste or type text containing causal statements…"
          spellCheck={false}
        />
      </div>

      <div className="analyzer-pane analyzer-pane--result">
        <div className="analyzer-pane-header">
          <span className="analyzer-pane-title">Analysis</span>
          {result && (
            <span className="analyzer-pane-meta">
              {result.relations.length} relation{result.relations.length !== 1 ? 's' : ''} found
            </span>
          )}
        </div>

        {isAnalyzing && <div className="loading">Analyzing…</div>}

        {result && !isAnalyzing && (
          <>
            {/* Highlighted text */}
            <div className="analyzer-highlighted-text">
              {renderHighlighted(result.text, result.spans)}
            </div>

            <div className="analyzer-legend">
              <span className="analyzer-legend-item">
                <mark className="span-cause">cause</mark>
              </span>
              <span className="analyzer-legend-item">
                <mark className="span-effect">effect</mark>
              </span>
            </div>

            {result.relations.length > 0 ? (
              <>
                <div className="cluster-section-label" style={{ marginTop: 16 }}>
                  Extracted relations
                </div>
                {result.relations.map((r, i) => (
                  <div key={i} className="analyzer-relation">
                    <span className="analyzer-relation-row">
                      <mark className="span-cause">{r.cause_text}</mark>
                      <span className="analyzer-relation-arrow">→</span>
                      <mark className="span-effect">{r.effect_text}</mark>
                    </span>
                    {r.cause_cluster_label && r.effect_cluster_label && (
                      <span className="analyzer-cluster-row">
                        <span className="tag">{r.cause_cluster_label}</span>
                        <span className="analyzer-relation-arrow">→</span>
                        <span className="tag">{r.effect_cluster_label}</span>
                        {r.corpus_post_count > 0 && (
                          <span className="analyzer-corpus-count">
                            · {r.corpus_post_count.toLocaleString()} posts in corpus
                          </span>
                        )}
                      </span>
                    )}
                  </div>
                ))}
              </>
            ) : (
              <p className="hint">No causal relations detected. Try sentences with words like "causes", "leads to", "increases", "reduces".</p>
            )}

            <p className="analyzer-mock-note">
              Mock extraction — connect backend endpoint <code>POST /api/analyze</code> for NLP-based analysis and corpus evidence counts.
            </p>
          </>
        )}

        {!result && !isAnalyzing && (
          <p className="hint">
            Causal spans will be highlighted and extracted cause→effect relationships will appear here.
          </p>
        )}
      </div>
    </div>
  )
}

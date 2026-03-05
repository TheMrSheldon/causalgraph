import { type FormEvent, useState } from 'react'
import { api } from '../api/client'
import type { PathsResponse } from '../types'
import { SankeyDiagram } from './SankeyDiagram'

export function PathFinderScreen() {
  const [causeQuery, setCauseQuery] = useState('')
  const [effectQuery, setEffectQuery] = useState('')
  const [result, setResult] = useState<PathsResponse | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [submitted, setSubmitted] = useState(false)

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault()
    if (!causeQuery.trim() || !effectQuery.trim()) return
    setIsLoading(true)
    setSubmitted(true)
    const data = await api.findPaths(causeQuery.trim(), effectQuery.trim())
    setResult(data)
    setIsLoading(false)
  }

  return (
    <div className="pathfinder-screen">
      <div className="screen-intro">
        <h2>Causal Path Finder</h2>
        <p>Enter a cause and an effect to find the causal pathways supported by the r/science corpus.</p>
      </div>

      <form className="pathfinder-form" onSubmit={handleSubmit}>
        <div className="pathfinder-query">
          <div className="pathfinder-field">
            <label className="pathfinder-label" htmlFor="cause-input">Cause</label>
            <input
              id="cause-input"
              className="pathfinder-input"
              value={causeQuery}
              onChange={e => setCauseQuery(e.target.value)}
              placeholder="e.g. climate change"
            />
          </div>
          <span className="pathfinder-arrow" aria-hidden="true">→</span>
          <div className="pathfinder-field">
            <label className="pathfinder-label" htmlFor="effect-input">Effect</label>
            <input
              id="effect-input"
              className="pathfinder-input"
              value={effectQuery}
              onChange={e => setEffectQuery(e.target.value)}
              placeholder="e.g. extreme weather"
            />
          </div>
          <button
            type="submit"
            className="pathfinder-submit"
            disabled={!causeQuery.trim() || !effectQuery.trim() || isLoading}
          >
            {isLoading ? 'Searching…' : 'Find paths'}
          </button>
        </div>
      </form>

      {isLoading && <div className="loading">Finding causal paths…</div>}

      {result && !isLoading && (
        result.nodes.length > 0
          ? <>
              <div className="sankey-note">
                Mock response — connect backend endpoint <code>GET /api/paths</code> for real corpus paths.
              </div>
              <SankeyDiagram nodes={result.nodes} links={result.links} />
            </>
          : <p className="hint">No causal paths found between these two concepts in the corpus.</p>
      )}

      {!submitted && (
        <p className="hint" style={{ marginTop: 32 }}>
          Results will show as a Sankey diagram with the number of supporting posts per path segment.
        </p>
      )}
    </div>
  )
}

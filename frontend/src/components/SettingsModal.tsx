import { useEffect, useRef, useState } from 'react'
import { useQueryClient } from '@tanstack/react-query'
import { setApiOverrides, getApiOverrides } from '../api/client'
import type { AnimationSpeed, GraphSettings, LayoutAlgorithm, NodeSpacing, VisualizationMode } from '../types'

// ---------------------------------------------------------------------------
// Endpoint validation + health-ping helpers
// ---------------------------------------------------------------------------

type PingStatus =
  | { kind: 'idle' }
  | { kind: 'checking' }
  | { kind: 'ok';    message: string }
  | { kind: 'error'; message: string }

function isValidHttpUrl(raw: string): boolean {
  try {
    const u = new URL(raw)
    return u.protocol === 'http:' || u.protocol === 'https:'
  } catch { return false }
}

async function pingHealth(baseUrl: string): Promise<PingStatus> {
  const healthUrl = `${baseUrl.replace(/\/$/, '')}/health`
  const ac = new AbortController()
  const tid = setTimeout(() => ac.abort(), 5000)
  try {
    const res = await fetch(healthUrl, { signal: ac.signal })
    clearTimeout(tid)
    if (!res.ok) return { kind: 'error', message: `HTTP ${res.status}` }
    const body = await res.json().catch(() => null)
    if (body?.status === 'ok') return { kind: 'ok', message: 'Connected' }
    return { kind: 'error', message: 'Unexpected response' }
  } catch (err) {
    clearTimeout(tid)
    if (err instanceof Error && err.name === 'AbortError')
      return { kind: 'error', message: 'Timed out (5 s)' }
    return { kind: 'error', message: 'Not reachable' }
  }
}

// ---------------------------------------------------------------------------
// EndpointRow sub-component
// ---------------------------------------------------------------------------

function EndpointRow({
  label, value, onChange, status, placeholder,
}: {
  label: string
  value: string
  onChange: (v: string) => void
  status: PingStatus
  placeholder: string
}) {
  const inputClass = [
    'url-input',
    status.kind === 'ok'    ? 'url-input--ok'    : '',
    status.kind === 'error' ? 'url-input--error' : '',
  ].filter(Boolean).join(' ')

  return (
    <div className="url-field">
      <div className="settings-group-label">{label}</div>
      <input
        className={inputClass}
        type="url"
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        spellCheck={false}
        autoComplete="off"
      />
      {status.kind !== 'idle' && (
        <span className={`url-status url-status--${status.kind}`}>
          {status.kind === 'checking' && 'Checking\u2026'}
          {status.kind === 'ok'       && `\u2713 ${status.message}`}
          {status.kind === 'error'    && `\u2717 ${status.message}`}
        </span>
      )}
    </div>
  )
}

interface SettingsModalProps {
  open: boolean
  settings: GraphSettings
  onSettingsChange: (s: GraphSettings) => void
  onClose: () => void
  minPostCount: number
  onMinPostCountChange: (count: number) => void
  activeScreen: string
}

function SegmentedControl<T extends string>({
  value, onChange, options,
}: {
  value: T
  onChange: (v: T) => void
  options: { value: T; label: string }[]
}) {
  return (
    <div className="seg-control">
      {options.map((opt) => (
        <button
          key={opt.value}
          className={`seg-btn${value === opt.value ? ' seg-btn-active' : ''}`}
          onClick={() => onChange(opt.value)}
        >
          {opt.label}
        </button>
      ))}
    </div>
  )
}

function Toggle({
  label, value, onChange,
}: {
  label: string
  value: boolean
  onChange: (v: boolean) => void
}) {
  return (
    <label className="toggle-row">
      <span className="toggle-label">{label}</span>
      <button
        className={`toggle-switch${value ? ' toggle-on' : ''}`}
        onClick={() => onChange(!value)}
        role="switch"
        aria-checked={value}
      >
        <span className="toggle-thumb" />
      </button>
    </label>
  )
}

const VIZ_OPTIONS: { value: VisualizationMode; label: string }[] = [
  { value: 'no',      label: 'No'      },
  { value: 'opacity', label: 'Opacity' },
  { value: 'size',    label: 'Size'    },
]

const SPACING_OPTIONS: { value: NodeSpacing; label: string }[] = [
  { value: 'tight',  label: 'Tight'  },
  { value: 'normal', label: 'Normal' },
  { value: 'spread', label: 'Spread' },
]

const ANIMATION_OPTIONS: { value: AnimationSpeed; label: string }[] = [
  { value: 'off',    label: 'Off'    },
  { value: 'fast',   label: 'Fast'   },
  { value: 'normal', label: 'Normal' },
  { value: 'slow',   label: 'Slow'   },
]

const LAYOUT_OPTIONS: { value: LayoutAlgorithm; label: string }[] = [
  { value: 'fcose',       label: 'fCoSE'       },
  { value: 'cose',        label: 'CoSE'        },
  { value: 'breadthfirst', label: 'Hierarchy'  },
  { value: 'concentric',  label: 'Concentric'  },
  { value: 'circle',      label: 'Circle'      },
]

export function SettingsModal({
  open, settings, onSettingsChange, onClose,
  minPostCount, onMinPostCountChange, activeScreen,
}: SettingsModalProps) {
  const isExplorer = activeScreen === 'explorer'
  const queryClient = useQueryClient()

  // ── Endpoint override state ──────────────────────────────────────────────
  const savedOverrides = getApiOverrides()
  const [backendUrl,  setBackendUrl]  = useState(savedOverrides.backendUrl)
  const [pipelineUrl, setPipelineUrl] = useState(savedOverrides.pipelineUrl)
  const [backendStatus,  setBackendStatus]  = useState<PingStatus>({ kind: 'idle' })
  const [pipelineStatus, setPipelineStatus] = useState<PingStatus>({ kind: 'idle' })
  const backendTimer  = useRef<ReturnType<typeof setTimeout> | null>(null)
  const pipelineTimer = useRef<ReturnType<typeof setTimeout> | null>(null)

  function applyAndPing(
    newBackend: string,
    newPipeline: string,
    field: 'backend' | 'pipeline',
    setStatus: (s: PingStatus) => void,
  ) {
    // Persist both values together
    try {
      localStorage.setItem('api-endpoints', JSON.stringify({ backendUrl: newBackend, pipelineUrl: newPipeline }))
    } catch { /* ignore */ }

    // Apply to axios interceptors + invalidate cached queries
    setApiOverrides(newBackend, newPipeline)
    queryClient.invalidateQueries()

    const urlToCheck = field === 'backend' ? newBackend : newPipeline
    if (!urlToCheck) { setStatus({ kind: 'idle' }); return }

    if (!isValidHttpUrl(urlToCheck)) {
      setStatus({ kind: 'error', message: 'Invalid URL' })
      return
    }

    setStatus({ kind: 'checking' })
    pingHealth(urlToCheck).then(setStatus)
  }

  function handleBackendChange(value: string) {
    setBackendUrl(value)
    setBackendStatus({ kind: 'idle' })
    if (backendTimer.current) clearTimeout(backendTimer.current)
    backendTimer.current = setTimeout(() => applyAndPing(value, pipelineUrl, 'backend', setBackendStatus), 800)
  }

  function handlePipelineChange(value: string) {
    setPipelineUrl(value)
    setPipelineStatus({ kind: 'idle' })
    if (pipelineTimer.current) clearTimeout(pipelineTimer.current)
    pipelineTimer.current = setTimeout(() => applyAndPing(backendUrl, value, 'pipeline', setPipelineStatus), 800)
  }

  // Kick off a ping for any pre-filled values when the modal opens
  useEffect(() => {
    if (!open) return
    if (savedOverrides.backendUrl && isValidHttpUrl(savedOverrides.backendUrl)) {
      setBackendStatus({ kind: 'checking' })
      pingHealth(savedOverrides.backendUrl).then(setBackendStatus)
    }
    if (savedOverrides.pipelineUrl && isValidHttpUrl(savedOverrides.pipelineUrl)) {
      setPipelineStatus({ kind: 'checking' })
      pingHealth(savedOverrides.pipelineUrl).then(setPipelineStatus)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  if (!open) return null

  const set = <K extends keyof GraphSettings>(key: K, value: GraphSettings[K]) =>
    onSettingsChange({ ...settings, [key]: value })

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-panel" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h2>Settings</h2>
          <button className="drawer-close" onClick={onClose} title="Close">✕</button>
        </div>
        <div className="modal-body">

          <div className="settings-section">
            <div className="settings-section-title">API Connections</div>
            <div className="settings-group url-field-group">
              <EndpointRow
                label="Backend URL"
                value={backendUrl}
                onChange={handleBackendChange}
                status={backendStatus}
                placeholder="Default (proxied) — e.g. http://localhost:8000"
              />
            </div>
            <div className="settings-group url-field-group">
              <EndpointRow
                label="Pipeline URL"
                value={pipelineUrl}
                onChange={handlePipelineChange}
                status={pipelineStatus}
                placeholder="Default (proxied) — e.g. http://localhost:8001"
              />
            </div>
            <p className="url-field-hint">Leave blank to use the default proxied paths. Changes take effect immediately.</p>
          </div>

          {isExplorer && (
            <>
              <div className="settings-section">
                <div className="settings-section-title">Graph Data</div>
                <div className="settings-group">
                  <div className="settings-group-label">Min posts per edge</div>
                  <input
                    className="settings-number"
                    type="number"
                    min={1}
                    max={9999}
                    value={minPostCount}
                    onChange={(e) => onMinPostCountChange(Math.max(1, Number(e.target.value)))}
                  />
                </div>
              </div>

              <div className="settings-section">
                <div className="settings-section-title">Visual Encoding</div>
                <div className="settings-group">
                  <div className="settings-group-label">Cluster size</div>
                  <SegmentedControl value={settings.clusterSizeMode} onChange={(v) => set('clusterSizeMode', v)} options={VIZ_OPTIONS} />
                </div>
                <div className="settings-group">
                  <div className="settings-group-label">Link strength</div>
                  <SegmentedControl value={settings.linkSizeMode} onChange={(v) => set('linkSizeMode', v)} options={VIZ_OPTIONS} />
                </div>
                <Toggle label="Edge labels" value={settings.showEdgeLabels} onChange={(v) => set('showEdgeLabels', v)} />
                <Toggle label="Member count in label" value={settings.showMemberCount} onChange={(v) => set('showMemberCount', v)} />
              </div>

              <div className="settings-section">
                <div className="settings-section-title">Interaction</div>
                <Toggle label="Dim unrelated on selection" value={settings.dimOnSelection} onChange={(v) => set('dimOnSelection', v)} />
                <Toggle label="Highlight neighbors on hover" value={settings.highlightOnHover} onChange={(v) => set('highlightOnHover', v)} />
              </div>

              <div className="settings-section">
                <div className="settings-section-title">Layout</div>
                <div className="settings-group">
                  <div className="settings-group-label">Algorithm</div>
                  <select
                    className="settings-select"
                    value={settings.layoutAlgorithm}
                    onChange={(e) => set('layoutAlgorithm', e.target.value as LayoutAlgorithm)}
                  >
                    {LAYOUT_OPTIONS.map((o) => (
                      <option key={o.value} value={o.value}>{o.label}</option>
                    ))}
                  </select>
                </div>
                <div className="settings-group">
                  <div className="settings-group-label">Node spacing</div>
                  <SegmentedControl value={settings.nodeSpacing} onChange={(v) => set('nodeSpacing', v)} options={SPACING_OPTIONS} />
                </div>
                <div className="settings-group">
                  <div className="settings-group-label">Animation speed</div>
                  <SegmentedControl value={settings.animationSpeed} onChange={(v) => set('animationSpeed', v)} options={ANIMATION_OPTIONS} />
                </div>
              </div>

              <div className="settings-section">
                <div className="settings-section-title">Display</div>
                <Toggle label="Show arrows" value={settings.showArrows} onChange={(v) => set('showArrows', v)} />
                <Toggle label="Show legend" value={settings.showLegend} onChange={(v) => set('showLegend', v)} />
                <Toggle label="Highlight event spans in posts" value={settings.showHighlightSpans} onChange={(v) => set('showHighlightSpans', v)} />
              </div>
            </>
          )}

        </div>
      </div>
    </div>
  )
}

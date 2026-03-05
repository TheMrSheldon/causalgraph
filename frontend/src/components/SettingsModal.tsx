import type { GraphSettings, NodeSpacing, VisualizationMode } from '../types'

interface SettingsModalProps {
  open: boolean
  settings: GraphSettings
  onSettingsChange: (s: GraphSettings) => void
  onClose: () => void
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

export function SettingsModal({ open, settings, onSettingsChange, onClose }: SettingsModalProps) {
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
              <div className="settings-group-label">Node spacing</div>
              <SegmentedControl value={settings.nodeSpacing} onChange={(v) => set('nodeSpacing', v)} options={SPACING_OPTIONS} />
            </div>
            <Toggle label="Animate layout" value={settings.animateLayout} onChange={(v) => set('animateLayout', v)} />
          </div>

          <div className="settings-section">
            <div className="settings-section-title">Display</div>
            <Toggle label="Show arrows" value={settings.showArrows} onChange={(v) => set('showArrows', v)} />
            <Toggle label="Show legend" value={settings.showLegend} onChange={(v) => set('showLegend', v)} />
          </div>

        </div>
      </div>
    </div>
  )
}

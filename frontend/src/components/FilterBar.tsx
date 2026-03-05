import { useLevels } from '../hooks/useGraph'

interface FilterBarProps {
  level: number
  minPostCount: number
  onLevelChange: (level: number) => void
  onMinPostCountChange: (count: number) => void
}

export function FilterBar({ level, minPostCount, onLevelChange, onMinPostCountChange }: FilterBarProps) {
  const { data: levels } = useLevels()

  return (
    <div className="filter-bar">
      <label>
        Hierarchy level:{' '}
        <select value={level} onChange={(e) => onLevelChange(Number(e.target.value))}>
          {levels?.levels.map((l) => (
            <option key={l} value={l}>
              Level {l} ({levels.counts[String(l)] ?? 0} clusters)
            </option>
          ))}
        </select>
      </label>

      <label>
        Min posts per edge:{' '}
        <input
          type="number"
          min={1}
          max={9999}
          value={minPostCount}
          onChange={(e) => onMinPostCountChange(Math.max(1, Number(e.target.value)))}
          style={{ width: 60 }}
        />
      </label>
    </div>
  )
}

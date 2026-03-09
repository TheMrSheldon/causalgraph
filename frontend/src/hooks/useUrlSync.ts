export type Screen = 'explorer' | 'pathfinder' | 'analyzer'

const SCREEN_PATHS: Record<Screen, string> = {
  explorer:   '/explorer',
  pathfinder: '/pathfinder',
  analyzer:   '/analyzer',
}

const PATH_TO_SCREEN: Record<string, Screen> = {
  '/':           'explorer',
  '/explorer':   'explorer',
  '/pathfinder': 'pathfinder',
  '/analyzer':   'analyzer',
}

export interface UrlState {
  screen:   Screen
  node:     number | null
  edge:     { source: number; target: number } | null
  expanded: number[]
  focus:    number[]
  post:     string | null
  backend:  string
  pipeline: string
}

export function readUrlState(): UrlState {
  const path   = window.location.pathname
  const screen = PATH_TO_SCREEN[path] ?? 'explorer'
  const params = new URLSearchParams(window.location.search)

  const nodeStr = params.get('node')
  const node    = nodeStr != null ? parseInt(nodeStr, 10) : null

  const edgeStr = params.get('edge')
  let edge: UrlState['edge'] = null
  if (edgeStr) {
    const parts = edgeStr.split(',')
    if (parts.length === 2) {
      const s = parseInt(parts[0], 10)
      const t = parseInt(parts[1], 10)
      if (!isNaN(s) && !isNaN(t)) edge = { source: s, target: t }
    }
  }

  const expStr  = params.get('exp')
  const expanded = expStr
    ? expStr.split(',').map(Number).filter((n) => !isNaN(n))
    : []

  const focusStr = params.get('focus')
  const focus    = focusStr
    ? focusStr.split(',').map(Number).filter((n) => !isNaN(n))
    : []

  const post     = params.get('post')
  const backend  = params.get('backend')  ?? ''
  const pipeline = params.get('pipeline') ?? ''

  return { screen, node, edge, expanded, focus, post, backend, pipeline }
}

export function syncUrlState(state: UrlState, replace = true): void {
  const path   = SCREEN_PATHS[state.screen]
  const params = new URLSearchParams()

  if (state.node != null)        params.set('node',     String(state.node))
  if (state.edge)                params.set('edge',     `${state.edge.source},${state.edge.target}`)
  if (state.expanded.length > 0) params.set('exp',      state.expanded.join(','))
  if (state.focus.length > 0)    params.set('focus',    state.focus.join(','))
  if (state.post)                params.set('post',     state.post)
  if (state.backend)             params.set('backend',  state.backend)
  if (state.pipeline)            params.set('pipeline', state.pipeline)

  const search  = params.toString() ? `?${params.toString()}` : ''
  const url     = `${path}${search}`
  const current = `${window.location.pathname}${window.location.search}`
  if (current === url) return

  if (replace) window.history.replaceState(null, '', url)
  else         window.history.pushState(null, '', url)
}

/**
 * Build a shareable URL for a specific post entry.
 *
 * includeGraphState=false → path + sidebar context (node= or edge=) + post=<id>
 *                           enough to open the correct detail view and highlight the post
 * includeGraphState=true  → full current URL with post= added/replaced
 *                           preserves expanded clusters, focus stack, endpoint overrides, etc.
 */
export function buildShareUrl(postId: string, includeGraphState: boolean): string {
  const origin  = window.location.origin
  const path    = window.location.pathname
  const current = new URLSearchParams(window.location.search)

  if (includeGraphState) {
    const params = new URLSearchParams(current)
    params.set('post', postId)
    return `${origin}${path}?${params.toString()}`
  }

  // "Post only": keep just the sidebar context + post; drop expanded/focus/backend/pipeline
  const minimal = new URLSearchParams()
  const node = current.get('node')
  const edge = current.get('edge')
  if (node) minimal.set('node', node)
  else if (edge) minimal.set('edge', edge)
  minimal.set('post', postId)
  return `${origin}${path}?${minimal.toString()}`
}

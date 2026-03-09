import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: process.env.VITE_BACKEND_URL || 'http://localhost:8000',
        changeOrigin: true,
      },
      '/pipeline': {
        target: process.env.VITE_PIPELINE_URL || 'http://localhost:8001',
        changeOrigin: true,
        rewrite: (path: string) => path.replace(/^\/pipeline/, ''),
      },
    },
  },
})

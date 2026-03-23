import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'

const apiPort = process.env.CLASSIFY_API_PORT || '8501'
const apiTarget = `http://localhost:${apiPort}`

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/demos/cvt/",
  server: {
    port: 5173,
    allowedHosts: ['iml.dfki.de'],
    watch: {
      usePolling: true,
      interval: 1000,
    },
    proxy: {
      '/demos/cvt/api': {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/demos\/cvt/, ''),
      },
      '/demos/cvt/samples': {
        target: apiTarget,
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/demos\/cvt/, ''),
      },
    },
  },
})

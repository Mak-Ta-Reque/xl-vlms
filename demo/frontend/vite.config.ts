import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'


// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/demos/cvt/",
  server: {
    port: 5173,
    allowedHosts: ['iml.dfki.de'],
    proxy: {
      '/demos/cvt/api': {
        target: 'http://localhost:8501',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/demos\/cvt/, ''),
      },
      '/demos/cvt/samples': {
        target: 'http://localhost:8501',
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/demos\/cvt/, ''),
      },
    },
  },
})

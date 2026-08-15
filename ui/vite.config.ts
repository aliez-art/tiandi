import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// 开发模式：后端服务跑在 18765（tiandi server）；生产（Tauri/静态部署）用绝对地址即可
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:18765',
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  },
})

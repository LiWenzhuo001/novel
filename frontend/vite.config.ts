import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import tailwindcss from 'tailwindcss'
import autoprefixer from 'autoprefixer'

const tailwindConfig = fileURLToPath(new URL('./tailwind.config.js', import.meta.url))

export default defineConfig({
  plugins: [vue()],
  // 内联 postcss 与 postcss.config.js 等价，作为双保险保留（此前 uni 管线不加载 postcss.config.js 的教训）
  css: {
    postcss: {
      plugins: [
        tailwindcss({ config: tailwindConfig }),
        autoprefixer,
      ],
    },
  },
  server: {
    host: '0.0.0.0',
    port: 5173,
    // 开发态把 /api 转发到后端（与 docker nginx 行为一致）
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
})

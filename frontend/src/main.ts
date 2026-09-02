import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import './style.css'

console.log('小说智读 RAG 启动')

createApp(App).use(router).mount('#app')

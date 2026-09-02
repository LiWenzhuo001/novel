<script setup lang="ts">
// 轻量内联 SVG 图标，替代 lucide-vue-next（避免 uni-app 编译/浏览器依赖问题）。
// 统一 24x24 视图、currentColor 描边，跟随父级文字颜色。
defineProps<{ name: string; size?: number | string }>()
</script>

<template>
  <svg
    :width="size || 18"
    :height="size || 18"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    stroke-width="2"
    stroke-linecap="round"
    stroke-linejoin="round"
    aria-hidden="true"
  >
    <!-- Sparkles -->
    <g v-if="name === 'sparkles'">
      <path d="M12 3l1.9 4.6L18.5 9.5l-4.6 1.9L12 16l-1.9-4.6L5.5 9.5l4.6-1.9z" />
      <path d="M19 14l.8 2 2 .8-2 .8-.8 2-.8-2-2-.8 2-.8z" />
    </g>
    <!-- FileText -->
    <g v-else-if="name === 'file-text'">
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
      <path d="M9 13h6M9 17h6" />
    </g>
    <!-- Send -->
    <g v-else-if="name === 'send'">
      <path d="M22 2L11 13" />
      <path d="M22 2l-7 20-4-9-9-4z" />
    </g>
    <!-- Bot -->
    <g v-else-if="name === 'bot'">
      <rect x="4" y="8" width="16" height="11" rx="2" />
      <path d="M12 8V4M9 4h6" />
      <circle cx="9" cy="13" r="1" fill="currentColor" />
      <circle cx="15" cy="13" r="1" fill="currentColor" />
    </g>
    <!-- User -->
    <g v-else-if="name === 'user'">
      <circle cx="12" cy="8" r="4" />
      <path d="M4 21a8 8 0 0 1 16 0" />
    </g>
    <!-- ChevronDown -->
    <g v-else-if="name === 'chevron-down'">
      <path d="M6 9l6 6 6-6" />
    </g>
    <!-- UploadCloud -->
    <g v-else-if="name === 'upload-cloud'">
      <path d="M16 16l-4-4-4 4" />
      <path d="M12 12v9" />
      <path d="M20.4 18.4A5 5 0 0 0 18 9h-1.3A8 8 0 1 0 3 16.3" />
    </g>
    <!-- RefreshCw -->
    <g v-else-if="name === 'refresh'">
      <path d="M20 11a8 8 0 0 0-14.8-4M4 4v5h5" />
      <path d="M4 13a8 8 0 0 0 14.8 4M20 20v-5h-5" />
    </g>    <!-- Trash2 -->
    <g v-else-if="name === 'trash'">
      <path d="M3 6h18" />
      <path d="M8 6V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
    </g>
    <!-- FolderOpen -->
    <g v-else-if="name === 'folder-open'">
      <path d="M3 7a2 2 0 0 1 2-2h4l2 3h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
    </g>
    <!-- Briefcase -->
    <g v-else-if="name === 'briefcase'">
      <rect x="3" y="7" width="18" height="13" rx="2" />
      <path d="M8 7V5a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <path d="M3 12h18" />
    </g>
    <!-- GraduationCap -->
    <g v-else-if="name === 'graduation'">
      <path d="M22 10L12 5 2 10l10 5 10-5z" />
      <path d="M6 12v5c0 1 2.7 2.5 6 2.5S18 18 18 17v-5" />
    </g>
    <!-- Download -->
    <g v-else-if="name === 'download'">
      <path d="M12 3v12" />
      <path d="M7 10l5 5 5-5" />
      <path d="M5 21h14" />
    </g>
    <!-- Loader (用旋转动画包裹) -->
    <g v-else-if="name === 'loader'" class="icon-spin">
      <path d="M12 2v4M12 18v4M4.9 4.9l2.8 2.8M16.3 16.3l2.8 2.8M2 12h4M18 12h4M4.9 19.1l2.8-2.8M16.3 7.7l2.8-2.8" />
    </g>
    <!-- X -->
    <g v-else-if="name === 'x'">
      <path d="M18 6L6 18M6 6l12 12" />
    </g>
    <!-- ArrowLeft -->
    <g v-else-if="name === 'arrow-left'">
      <path d="M19 12H5M12 19l-7-7 7-7" />
    </g>
    <!-- MessagesSquare -->
    <g v-else-if="name === 'messages'">
      <path d="M14 9a2 2 0 0 1-2 2H6l-4 4V4a2 2 0 0 1 2-2h8a2 2 0 0 1 2 2z" />
      <path d="M18 9h2a2 2 0 0 1 2 2v11l-4-4h-6a2 2 0 0 1-2-2v-1" />
    </g>
    <!-- Lightbulb -->
    <g v-else-if="name === 'lightbulb'">
      <path d="M9 18h6M10 21h4" />
      <path d="M12 3a6 6 0 0 0-4 10.5c.7.7 1 1.5 1 2.5h6c0-1 .3-1.8 1-2.5A6 6 0 0 0 12 3z" />
    </g>
    <!-- Check -->
    <g v-else-if="name === 'check'">
      <path d="M20 6L9 17l-5-5" />
    </g>
    <!-- Copy -->
    <g v-else-if="name === 'copy'">
      <rect x="9" y="9" width="13" height="13" rx="2" />
      <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
    </g>
    <!-- Wrench (工具) -->
    <g v-else-if="name === 'wrench'">
      <path d="M14.7 6.3a4.5 4.5 0 0 0-6 6L3 18l3 3 5.7-5.7a4.5 4.5 0 0 0 6-6L14 13l-3-3z" />
    </g>
    <!-- LogOut -->
    <g v-else-if="name === 'logout'">
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
      <path d="M16 17l5-5-5-5" />
      <path d="M21 12H9" />
    </g>
    <!-- 兜底 -->
    <circle v-else cx="12" cy="12" r="9" />
  </svg>
</template>

<style scoped>
.icon-spin {
  transform-origin: center;
  animation: icon-spin 1s linear infinite;
}
@keyframes icon-spin {
  to {
    transform: rotate(360deg);
  }
}
</style>

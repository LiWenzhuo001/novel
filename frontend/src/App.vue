<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'
import AppLayout from './components/AppLayout.vue'

const route = useRoute()
// 登录页不需要应用壳；其余页面统一由 Layout 提供侧栏与顶栏。
const bare = computed(() => Boolean(route.meta.public))
</script>

<template>
  <div class="h-full">
    <AppLayout v-if="!bare">
      <router-view />
    </AppLayout>
    <router-view v-else />
  </div>
</template>

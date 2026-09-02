import js from '@eslint/js'
import pluginVue from 'eslint-plugin-vue'
import { defineConfigWithVueTs, vueTsConfigs } from '@vue/eslint-config-typescript'

// 首期只开"真实错误"级规则（与后端 ruff F+E9 的保守策略一致），
// 格式类模板规则关闭，留给将来引入 Prettier 时统一处理。
export default defineConfigWithVueTs(
  { ignores: ['dist/**', 'node_modules/**'] },
  js.configs.recommended,
  ...pluginVue.configs['flat/recommended'],
  vueTsConfigs.recommended,
  {
    rules: {
      'vue/max-attributes-per-line': 'off',
      'vue/singleline-html-element-content-newline': 'off',
      'vue/html-self-closing': 'off',
      'vue/html-indent': 'off',
      'vue/attributes-order': 'off',
      'vue/first-attribute-linebreak': 'off',
      'vue/html-closing-bracket-newline': 'off',
      // v-html 的内容一律经 DOMPurify.sanitize 后渲染（ChatPanel.vue:153）。
      'vue/no-v-html': 'off',
      '@typescript-eslint/no-explicit-any': 'off',
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_' }],
    },
  },
  {
    // 路由页面与单字基础组件按文件名单词命名是本项目的既有惯例。
    files: ['src/pages/**/*.vue', 'src/components/Icon.vue'],
    rules: { 'vue/multi-word-component-names': 'off' },
  },
)

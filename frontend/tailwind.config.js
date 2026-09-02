// Tailwind CSS 主题配置——uni-app H5 前端设计系统
//
// 自定义内容：
// - 品牌色系（brand: indigo + violet 渐变）、文本色系（ink: 多层级灰）
// - 输入框背景色（surface）、页面背景（paper）
// - 4 组投影（card / pop / glow / doc）
// - 5 组动画（fadeIn / floaty / gradientX / pulseSoft / caret）
// - 中英文混合字体栈（Inter + 思源黑体 / 苹方 / 微软雅黑）

import tailwindcssAnimate from 'tailwindcss-animate'

/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{vue,js,ts,jsx,tsx}'],
  theme: {
    extend: {
      fontFamily: {
        sans: ['"Inter"', '"Source Han Sans SC"', '"Noto Sans SC"', '"PingFang SC"', '"Microsoft YaHei"', 'system-ui', 'sans-serif'],
        display: ['"Noto Serif SC"', '"Source Han Serif SC"', '"Songti SC"', 'STSong', 'SimSun', 'serif'],
      },
      colors: {
        brand: {
          50: '#EEF2FF',
          100: '#E0E7FF',
          200: '#C7D2FE',
          300: '#A5B4FC',
          400: '#818CF8',
          500: '#6366F1',
          600: '#4F46E5',
          700: '#4338CA',
          800: '#3730A3',
          900: '#312E81',
        },
        violet: {
          500: '#8B5CF6',
          600: '#7C3AED',
        },
        ink: {
          DEFAULT: '#17171C',
          soft: '#3F3F46',
          mute: '#71717A',
          faint: '#A1A1AA',
        },
        paper: '#F7F6F3',
      },
      boxShadow: {
        card: '0 1px 2px rgba(23,23,28,0.04), 0 8px 24px -12px rgba(23,23,28,0.12)',
        pop: '0 2px 8px rgba(23,23,28,0.06), 0 16px 48px -16px rgba(23,23,28,0.18)',
        glow: '0 0 0 1px rgba(99,102,241,0.25), 0 8px 32px -8px rgba(99,102,241,0.45)',
        doc: '0 1px 3px rgba(23,23,28,0.06), 0 24px 64px -24px rgba(23,23,28,0.25)',
      },
      borderRadius: {
        xl2: '1.25rem',
      },
      keyframes: {
        fadeIn: {
          '0%': { opacity: '0', transform: 'translateY(10px)' },
          '100%': { opacity: '1', transform: 'translateY(0)' },
        },
        floaty: {
          '0%,100%': { transform: 'translateY(0)' },
          '50%': { transform: 'translateY(-8px)' },
        },
        gradientX: {
          '0%,100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
        pulseSoft: {
          '0%,100%': { opacity: '1', transform: 'scale(1)' },
          '50%': { opacity: '0.75', transform: 'scale(0.96)' },
        },
        caret: {
          '0%,100%': { opacity: '1' },
          '50%': { opacity: '0' },
        },
      },
      animation: {
        fadeIn: 'fadeIn .45s cubic-bezier(.22,.68,.36,1) both',
        floaty: 'floaty 6s ease-in-out infinite',
        gradientX: 'gradientX 8s ease infinite',
        pulseSoft: 'pulseSoft 2.4s ease-in-out infinite',
        caret: 'caret 1s step-end infinite',
      },
    },
  },
  plugins: [tailwindcssAnimate],
}

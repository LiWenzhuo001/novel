// 零依赖轻提示与剪贴板工具（替代 uni.showToast / uni.setClipboardData）
// 样式在 style.css 中手写（Tailwind JIT 扫不到运行时创建的 DOM）

let timer: number | undefined

export function showToast(title: string, duration = 2200) {
  let el = document.querySelector<HTMLDivElement>('.app-toast')
  if (!el) {
    el = document.createElement('div')
    el.className = 'app-toast'
    el.setAttribute('role', 'status')
    el.setAttribute('aria-live', 'polite')
    el.setAttribute('aria-atomic', 'true')
    document.body.appendChild(el)
  }
  el.textContent = title
  el.classList.add('app-toast--show')
  window.clearTimeout(timer)
  timer = window.setTimeout(() => el!.classList.remove('app-toast--show'), duration)
}

export async function copyText(text: string): Promise<void> {
  try {
    await navigator.clipboard.writeText(text)
    return
  } catch {
    // 非安全上下文（http/非 localhost）回退 execCommand
  }
  const ta = document.createElement('textarea')
  ta.value = text
  ta.style.cssText = 'position:fixed;opacity:0;pointer-events:none'
  document.body.appendChild(ta)
  ta.select()
  document.execCommand('copy')
  document.body.removeChild(ta)
}

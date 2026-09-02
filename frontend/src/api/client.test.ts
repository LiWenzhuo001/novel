import { describe, expect, it } from 'vitest'

import { parseSseBlock } from './client'

describe('parseSseBlock', () => {
  it('解析单行 data 的标准块', () => {
    const block = 'event: token\ndata: 你好'
    expect(parseSseBlock(block)).toEqual({ event: 'token', data: '你好' })
  })

  it('多行 data 用 \\n 重连，保留 token 自身换行', () => {
    const block = 'event: token\ndata: 第一行\ndata: 第二行'
    expect(parseSseBlock(block)).toEqual({ event: 'token', data: '第一行\n第二行' })
  })

  it('容忍 \\r\\n 行尾（sse-starlette 服务端实际输出）', () => {
    const block = 'event: meta\r\ndata: {"strategy":"direct"}\r\n'
    expect(parseSseBlock(block)).toEqual({ event: 'meta', data: '{"strategy":"direct"}' })
  })

  it('data: 后至多剥一个前导空格，保留正文空格', () => {
    const block = 'event: token\ndata:  保留两个空格'
    expect(parseSseBlock(block)).toEqual({ event: 'token', data: ' 保留两个空格' })
  })

  it('无 data 的块返回空字符串', () => {
    expect(parseSseBlock('event: done')).toEqual({ event: 'done', data: '' })
  })

  it('JSON payload 完整保留（含冒号与花括号）', () => {
    const payload = '{"route": "a:b", "nested": {"k": 1}}'
    const block = `event: route\ndata: ${payload}`
    expect(parseSseBlock(block).data).toBe(payload)
  })
})

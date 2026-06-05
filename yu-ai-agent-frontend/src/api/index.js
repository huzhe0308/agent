import axios from 'axios'

// 开发环境通过 vite proxy 转发到 8123，生产环境使用同域 /api
const API_BASE_URL = '/api'

const request = axios.create({
  baseURL: API_BASE_URL,
  timeout: 60000,
})

/** SSE 流式连接 */
export const connectSSE = (url, params, onMessage, onError) => {
  const queryString = Object.keys(params)
    .filter((k) => params[k] !== undefined && params[k] !== null && params[k] !== '')
    .map((key) => `${encodeURIComponent(key)}=${encodeURIComponent(params[key])}`)
    .join('&')

  const fullUrl = `${API_BASE_URL}${url}?${queryString}`
  const eventSource = new EventSource(fullUrl)

  eventSource.onmessage = (event) => {
    const data = event.data
    if (data === '[DONE]') {
      onMessage?.('[DONE]')
    } else {
      onMessage?.(data)
    }
  }

  eventSource.onerror = (error) => {
    onError?.(error)
    eventSource.close()
  }

  return eventSource
}

// ---------- 金融理财咨询 ----------
export const chatWithFinAdvisor = (message, chatId, riskProfile) =>
  connectSSE('/ai/fin_advisor/chat/sse', { message, chatId, riskProfile })

export const chatWithFinAdvisorSync = (message, chatId, riskProfile) =>
  request.post('/ai/fin_advisor/chat', {
    message,
    chatId,
    risk_profile: riskProfile,
  })

// ---------- FinManus 超级智能体 ----------
export const chatWithFinManus = (message, chatId = 'manus_default') =>
  connectSSE('/ai/manus/chat', { message, chatId })

// ---------- Harness 管理 ----------
export const getHealth = () => axios.get('/health')

export const getSession = (chatId) => request.get(`/ai/fin_advisor/session/${chatId}`)

export const clearSession = (chatId) => request.delete(`/ai/fin_advisor/session/${chatId}`)

export const getMemoryContext = (chatId) => request.get(`/harness/memory/${chatId}`)

export const listRuns = (chatId) =>
  request.get('/harness/runs', { params: chatId ? { chatId } : {} })

export const getPendingApprovals = (sessionId) =>
  request.get('/harness/approvals/pending', { params: sessionId ? { sessionId } : {} })

export const approveToolCall = (approvalId) =>
  request.post(`/harness/approvals/${approvalId}/approve`)

export default {
  chatWithFinAdvisor,
  chatWithFinManus,
  getHealth,
  getSession,
  clearSession,
  getMemoryContext,
  listRuns,
  getPendingApprovals,
  approveToolCall,
}

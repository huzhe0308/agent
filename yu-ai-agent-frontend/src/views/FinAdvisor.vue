<template>
  <div class="page">
    <header class="header">
      <button class="back" @click="router.push('/')">← 返回</button>
      <div class="header-center">
        <h1>理财顾问</h1>
        <span class="session-tag">会话 {{ chatId }}</span>
      </div>
      <button class="clear-btn" @click="clearChat" title="清除会话">清除</button>
    </header>

    <div class="toolbar">
      <span class="toolbar-label">风险偏好</span>
      <button
        v-for="p in riskProfiles"
        :key="p.value"
        class="risk-chip"
        :class="{ active: riskProfile === p.value }"
        @click="riskProfile = p.value"
      >{{ p.label }}</button>
    </div>

    <div class="quick-prompts">
      <button
        v-for="q in quickPrompts"
        :key="q"
        class="prompt-chip"
        :disabled="connectionStatus === 'connecting'"
        @click="sendQuick(q)"
      >{{ q }}</button>
    </div>

    <div class="chat-wrap">
      <ChatRoom
        :messages="messages"
        :connection-status="connectionStatus"
        ai-type="advisor"
        theme="advisor"
        placeholder="请输入理财问题，如：基金定投10年收益多少？"
        @send-message="sendMessage"
      />
    </div>

    <AppFooter />
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount } from 'vue'
import { useRouter } from 'vue-router'
import { useHead } from '@vueuse/head'
import ChatRoom from '../components/ChatRoom.vue'
import AppFooter from '../components/AppFooter.vue'
import { chatWithFinAdvisor, clearSession } from '../api'

useHead({
  title: '理财顾问 - FinAdvisor',
  meta: [{ name: 'description', content: '金融理财咨询助手，RAG 知识问答与数值测算' }],
})

const router = useRouter()
const messages = ref([])
const chatId = ref('')
const connectionStatus = ref('disconnected')
const riskProfile = ref('稳健')
let eventSource = null

const riskProfiles = [
  { label: '保守', value: '保守' },
  { label: '稳健', value: '稳健' },
  { label: '积极', value: '积极' },
]

const quickPrompts = [
  '什么是基金定投？',
  '帮我算一下每月2000元定投10年收益',
  '保守型投资者适合什么产品？',
  '写一份理财配置分析报告',
]

const generateChatId = () => 'fin_' + Math.random().toString(36).substring(2, 10)

const addMessage = (content, isUser) => {
  messages.value.push({ content, isUser, time: Date.now() })
}

const sendMessage = (message) => {
  if (!message.trim()) return
  addMessage(message, true)

  if (eventSource) eventSource.close()

  const aiIdx = messages.value.length
  addMessage('', false)
  connectionStatus.value = 'connecting'

  eventSource = chatWithFinAdvisor(message, chatId.value, riskProfile.value)

  eventSource.onmessage = (event) => {
    const data = event.data
    if (data && data !== '[DONE]') {
      if (aiIdx < messages.value.length) {
        messages.value[aiIdx].content += data
      }
    }
    if (data === '[DONE]') {
      connectionStatus.value = 'disconnected'
      eventSource.close()
    }
  }

  eventSource.onerror = () => {
    connectionStatus.value = 'error'
    eventSource.close()
    if (!messages.value[aiIdx].content) {
      messages.value[aiIdx].content = '连接失败，请确认后端服务已启动（端口 8123）。'
    }
  }
}

const sendQuick = (q) => sendMessage(q)

const clearChat = async () => {
  try {
    await clearSession(chatId.value)
  } catch (_) { /* ignore */ }
  messages.value = []
  chatId.value = generateChatId()
  addMessage('会话已重置。我是您的理财顾问助手，可解答理财知识、测算收益、生成投资报告。请问有什么可以帮您？', false)
}

onMounted(() => {
  chatId.value = generateChatId()
  addMessage(
    '您好，我是 FinAdvisor 理财顾问。\n\n我可以帮您：\n• 解答理财知识与产品问题\n• 测算复利、定投收益\n• 评估风险偏好\n• 生成投资分析报告\n\n请选择上方风险偏好后开始咨询。',
    false,
  )
})

onBeforeUnmount(() => {
  eventSource?.close()
})
</script>

<style scoped>
.page {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
  background: #f4f7fa;
}

.header {
  display: grid;
  grid-template-columns: 100px 1fr 80px;
  align-items: center;
  padding: 14px 24px;
  background: linear-gradient(90deg, #0a1628, #1a3a5c);
  color: #fff;
  position: sticky;
  top: 0;
  z-index: 10;
}

.back, .clear-btn {
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(255,255,255,0.2);
  color: #fff;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
}
.back:hover, .clear-btn:hover { background: rgba(255,255,255,0.2); }

.header-center { text-align: center; }
.header-center h1 { font-size: 1.2rem; font-weight: 600; }
.session-tag { font-size: 0.75rem; opacity: 0.6; }

.toolbar {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 12px 24px;
  background: #fff;
  border-bottom: 1px solid #e8edf2;
  flex-wrap: wrap;
}

.toolbar-label { font-size: 0.85rem; color: #5a6a7e; }

.risk-chip {
  padding: 5px 16px;
  border-radius: 20px;
  border: 1px solid #d0d8e4;
  background: #fff;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s;
}
.risk-chip.active {
  background: #1a7f6e;
  color: #fff;
  border-color: #1a7f6e;
}

.quick-prompts {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 10px 24px;
  background: #fff;
  border-bottom: 1px solid #e8edf2;
}

.prompt-chip {
  font-size: 0.8rem;
  padding: 6px 12px;
  border-radius: 16px;
  border: 1px solid #d0d8e4;
  background: #f8fafc;
  color: #3a4a5e;
  cursor: pointer;
}
.prompt-chip:hover:not(:disabled) { border-color: #1a7f6e; color: #1a7f6e; }
.prompt-chip:disabled { opacity: 0.5; cursor: not-allowed; }

.chat-wrap {
  flex: 1;
  padding: 16px 24px;
  max-width: 900px;
  width: 100%;
  margin: 0 auto;
}

@media (max-width: 600px) {
  .header { grid-template-columns: 70px 1fr 60px; padding: 10px 12px; }
  .session-tag { display: none; }
  .chat-wrap { padding: 12px; }
}
</style>

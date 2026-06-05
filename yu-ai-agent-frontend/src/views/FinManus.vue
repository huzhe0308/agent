<template>
  <div class="page">
    <header class="header">
      <button class="back" @click="router.push('/')">← 返回</button>
      <div class="header-center">
        <h1>FinManus 超级智能体</h1>
        <span class="badge">ReAct · Tool Calling</span>
      </div>
      <div class="placeholder"></div>
    </header>

    <div class="info-bar">
      <span>🛠️ 支持复利计算、知识检索、HTTP 数据、子 Agent 委托等 7 类工具</span>
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
        ai-type="manus"
        theme="manus"
        placeholder="描述您的复杂理财任务，智能体将自主推理并调用工具..."
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
import { chatWithFinManus } from '../api'

useHead({
  title: 'FinManus 超级智能体 - FinAdvisor',
  meta: [{ name: 'description', content: 'ReAct 金融超级智能体，自主工具调用与多步推理' }],
})

const router = useRouter()
const messages = ref([])
const chatId = ref('manus_' + Math.random().toString(36).substring(2, 10))
const connectionStatus = ref('disconnected')
let eventSource = null

const quickPrompts = [
  '检索货币基金相关知识并总结',
  '本金10万年化5%复利20年多少钱',
  '帮我分析稳健型用户的资产配置方案',
]

const addMessage = (content, isUser, type = '') => {
  messages.value.push({ content, isUser, type, time: Date.now() })
}

const sendMessage = (message) => {
  if (!message.trim()) return
  addMessage(message, true, 'user-question')

  if (eventSource) eventSource.close()
  connectionStatus.value = 'connecting'

  let buffer = []
  let aiIdx = messages.value.length
  addMessage('', false, 'ai-answer')

  eventSource = chatWithFinManus(message, chatId.value)

  eventSource.onmessage = (event) => {
    const data = event.data
    if (data && data !== '[DONE]') {
      buffer.push(data)
      if (aiIdx < messages.value.length) {
        messages.value[aiIdx].content = buffer.join('')
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
    if (aiIdx < messages.value.length && !messages.value[aiIdx].content) {
      messages.value[aiIdx].content = '连接失败，请确认后端服务已启动。'
    }
  }
}

const sendQuick = (q) => sendMessage(q)

onMounted(() => {
  addMessage(
    '你好，我是 FinManus 金融超级智能体。\n\n我具备 ReAct 推理能力，可以自主决定：\n• 调用金融计算工具\n• 检索理财知识库\n• 多步分析复杂问题\n\n请描述您的需求。',
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
  background: #faf8f4;
}

.header {
  display: grid;
  grid-template-columns: 100px 1fr 100px;
  align-items: center;
  padding: 14px 24px;
  background: linear-gradient(90deg, #1a1408, #3d3010);
  color: #fff;
  position: sticky;
  top: 0;
  z-index: 10;
}

.back {
  background: rgba(255,255,255,0.1);
  border: 1px solid rgba(201,162,39,0.3);
  color: #fff;
  padding: 6px 12px;
  border-radius: 6px;
  cursor: pointer;
  font-size: 0.85rem;
}
.back:hover { background: rgba(201,162,39,0.2); }

.header-center { text-align: center; }
.header-center h1 { font-size: 1.2rem; font-weight: 600; }
.badge {
  font-size: 0.7rem;
  color: #c9a227;
  letter-spacing: 1px;
}

.info-bar {
  padding: 10px 24px;
  background: #fff8e6;
  border-bottom: 1px solid #f0e4c0;
  font-size: 0.82rem;
  color: #6a5a20;
  text-align: center;
}

.quick-prompts {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  padding: 10px 24px;
  background: #fff;
  border-bottom: 1px solid #f0e4c0;
}

.prompt-chip {
  font-size: 0.8rem;
  padding: 6px 12px;
  border-radius: 16px;
  border: 1px solid #e8d9a8;
  background: #fffdf5;
  color: #5a4a18;
  cursor: pointer;
}
.prompt-chip:hover:not(:disabled) { border-color: #c9a227; }
.prompt-chip:disabled { opacity: 0.5; }

.chat-wrap {
  flex: 1;
  padding: 16px 24px;
  max-width: 900px;
  width: 100%;
  margin: 0 auto;
}
</style>

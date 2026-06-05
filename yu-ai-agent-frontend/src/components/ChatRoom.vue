<template>
  <div class="chat-container" :class="theme">
    <div class="chat-messages" ref="messagesContainer">
      <div v-for="(msg, index) in messages" :key="index" class="message-wrapper">
        <div v-if="!msg.isUser" class="message ai-message" :class="[msg.type]">
          <div class="avatar ai-avatar">
            <AiAvatarFallback :type="aiType" />
          </div>
          <div class="message-bubble">
            <div class="message-content">{{ msg.content }}<span v-if="connectionStatus === 'connecting' && index === messages.length - 1" class="cursor">▋</span></div>
            <div class="message-time">{{ formatTime(msg.time) }}</div>
          </div>
        </div>

        <div v-else class="message user-message">
          <div class="message-bubble">
            <div class="message-content">{{ msg.content }}</div>
            <div class="message-time">{{ formatTime(msg.time) }}</div>
          </div>
          <div class="avatar user-avatar">
            <div class="avatar-placeholder">我</div>
          </div>
        </div>
      </div>

      <div v-if="connectionStatus === 'connecting'" class="status-hint">
        <span class="dot-pulse"></span> AI 正在思考...
      </div>
    </div>

    <div class="chat-input-container">
      <div class="chat-input">
        <textarea
          v-model="inputMessage"
          @keydown.enter.exact.prevent="sendMessage"
          :placeholder="placeholder"
          class="input-box"
          :disabled="connectionStatus === 'connecting'"
          rows="1"
        />
        <button
          class="send-button"
          :disabled="connectionStatus === 'connecting' || !inputMessage.trim()"
          @click="sendMessage"
        >发送</button>
      </div>
      <p class="disclaimer">以上内容仅供参考，不构成投资建议</p>
    </div>
  </div>
</template>

<script setup>
import { ref, watch, onMounted, nextTick } from 'vue'
import AiAvatarFallback from './AiAvatarFallback.vue'

const props = defineProps({
  messages: { type: Array, default: () => [] },
  connectionStatus: { type: String, default: 'disconnected' },
  aiType: { type: String, default: 'advisor' },
  theme: { type: String, default: 'advisor' },
  placeholder: { type: String, default: '请输入消息...' },
})

const emit = defineEmits(['send-message'])
const inputMessage = ref('')
const messagesContainer = ref(null)

const sendMessage = () => {
  if (!inputMessage.value.trim()) return
  emit('send-message', inputMessage.value)
  inputMessage.value = ''
}

const formatTime = (ts) =>
  new Date(ts).toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' })

const scrollToBottom = async () => {
  await nextTick()
  if (messagesContainer.value) {
    messagesContainer.value.scrollTop = messagesContainer.value.scrollHeight
  }
}

watch(() => props.messages.length, scrollToBottom)
watch(() => props.messages.map((m) => m.content).join(''), scrollToBottom)
onMounted(scrollToBottom)
</script>

<style scoped>
.chat-container {
  display: flex;
  flex-direction: column;
  height: calc(100vh - 280px);
  min-height: 480px;
  background: #fff;
  border-radius: 12px;
  border: 1px solid #e8edf2;
  overflow: hidden;
  position: relative;
  box-shadow: 0 4px 20px rgba(10, 22, 40, 0.06);
}

.chat-container.manus {
  border-color: #f0e4c0;
}

.chat-messages {
  flex: 1;
  overflow-y: auto;
  padding: 20px;
  padding-bottom: 100px;
}

.message-wrapper { margin-bottom: 16px; }

.message {
  display: flex;
  align-items: flex-start;
  max-width: 88%;
}

.user-message { margin-left: auto; }
.ai-message { margin-right: auto; }

.avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  flex-shrink: 0;
  overflow: hidden;
}

.ai-avatar { margin-right: 10px; }
.user-avatar { margin-left: 10px; }

.avatar-placeholder {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  background: #0a1628;
  color: #fff;
  font-size: 0.8rem;
  font-weight: 600;
}

.message-bubble {
  padding: 12px 16px;
  border-radius: 16px;
  word-wrap: break-word;
}

.advisor .user-message .message-bubble {
  background: #1a7f6e;
  color: #fff;
  border-bottom-right-radius: 4px;
}

.advisor .ai-message .message-bubble {
  background: #f0f4f8;
  color: #1a2332;
  border-bottom-left-radius: 4px;
}

.manus .user-message .message-bubble {
  background: #8a6d1a;
  color: #fff;
  border-bottom-right-radius: 4px;
}

.manus .ai-message .message-bubble {
  background: #faf6ec;
  color: #1a2332;
  border-bottom-left-radius: 4px;
}

.message-content {
  font-size: 0.95rem;
  line-height: 1.65;
  white-space: pre-wrap;
}

.message-time {
  font-size: 0.7rem;
  opacity: 0.55;
  margin-top: 6px;
  text-align: right;
}

.cursor {
  animation: blink 0.7s infinite;
  color: #1a7f6e;
}

@keyframes blink {
  0%, 100% { opacity: 0; }
  50% { opacity: 1; }
}

.status-hint {
  text-align: center;
  font-size: 0.85rem;
  color: #8a9bb0;
  padding: 8px;
}

.dot-pulse {
  display: inline-block;
  width: 8px;
  height: 8px;
  background: #1a7f6e;
  border-radius: 50%;
  margin-right: 6px;
  animation: pulse 1s infinite;
}

@keyframes pulse {
  0%, 100% { opacity: 0.3; transform: scale(0.8); }
  50% { opacity: 1; transform: scale(1); }
}

.chat-input-container {
  position: absolute;
  bottom: 0;
  left: 0;
  right: 0;
  background: #fff;
  border-top: 1px solid #e8edf2;
  padding: 12px 16px 8px;
}

.chat-input {
  display: flex;
  align-items: center;
  gap: 10px;
}

.input-box {
  flex: 1;
  border: 1px solid #d0d8e4;
  border-radius: 10px;
  padding: 10px 14px;
  font-size: 0.95rem;
  resize: none;
  outline: none;
  font-family: inherit;
  max-height: 80px;
}
.input-box:focus { border-color: #1a7f6e; }

.send-button {
  background: #0a1628;
  color: #fff;
  border: none;
  border-radius: 10px;
  padding: 10px 22px;
  font-size: 0.9rem;
  cursor: pointer;
  white-space: nowrap;
}
.send-button:hover:not(:disabled) { opacity: 0.9; }
.send-button:disabled { opacity: 0.45; cursor: not-allowed; }

.manus .send-button { background: #8a6d1a; }
.manus .input-box:focus { border-color: #c9a227; }

.disclaimer {
  text-align: center;
  font-size: 0.7rem;
  color: #a0aec0;
  margin-top: 6px;
}

@media (max-width: 600px) {
  .chat-container { height: calc(100vh - 320px); min-height: 400px; }
  .message { max-width: 95%; }
}
</style>

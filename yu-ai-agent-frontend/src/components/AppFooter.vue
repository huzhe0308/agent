<template>
  <footer class="footer">
    <div class="footer-inner">
      <div class="brand">
        <strong>FinAdvisor</strong>
        <span>金融理财智能平台</span>
      </div>
      <div class="links">
        <span>FastAPI · LangGraph · RAG · ReAct</span>
        <span class="sep">|</span>
        <span>Agent Harness · 四层记忆 · Checkpoint</span>
      </div>
    </div>
    <p class="copy">© {{ year }} FinAdvisor — 以上内容仅供参考，不构成投资建议</p>
  </footer>
</template>

<script setup>
import { computed } from 'vue'
const year = computed(() => new Date().getFullYear())
</script>

<style scoped>
.footer {
  background: #0a1628;
  color: rgba(255, 255, 255, 0.55);
  padding: 28px 24px 20px;
  margin-top: auto;
}

.footer-inner {
  max-width: 900px;
  margin: 0 auto;
  display: flex;
  flex-wrap: wrap;
  justify-content: space-between;
  align-items: center;
  gap: 12px;
  margin-bottom: 12px;
}

.brand strong {
  color: #c9a227;
  font-size: 1rem;
  margin-right: 10px;
}

.brand span { font-size: 0.85rem; }

.links {
  font-size: 0.78rem;
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  align-items: center;
}

.sep { opacity: 0.3; }

.copy {
  text-align: center;
  font-size: 0.75rem;
  opacity: 0.45;
}
</style>

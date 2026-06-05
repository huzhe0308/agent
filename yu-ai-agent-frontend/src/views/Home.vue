<template>
  <div class="home">
    <header class="hero">
      <div class="hero-badge">Enterprise AI Platform</div>
      <h1 class="hero-title">FinAdvisor</h1>
      <p class="hero-sub">金融理财智能平台</p>
      <p class="hero-desc">
        基于 LangGraph + RAG + ReAct 的企业级理财咨询系统<br />
        知识问答 · 数值测算 · 投资报告 · 工具调用
      </p>
    </header>

    <section class="features">
      <div class="feature-card" v-for="f in features" :key="f.title">
        <span class="feature-icon">{{ f.icon }}</span>
        <h3>{{ f.title }}</h3>
        <p>{{ f.desc }}</p>
      </div>
    </section>

    <section class="apps">
      <div class="app-card advisor" @click="go('/advisor')">
        <div class="app-icon">📊</div>
        <div class="app-body">
          <h2>理财顾问</h2>
          <p>意图识别 → 问题重写 → RAG 检索 → 智能生成</p>
          <ul>
            <li>理财知识 / 产品 / 监管政策问答</li>
            <li>复利、定投、风险偏好评分</li>
            <li>Markdown 投资分析报告</li>
          </ul>
        </div>
        <button class="app-btn">开始咨询 →</button>
      </div>

      <div class="app-card manus" @click="go('/manus')">
        <div class="app-icon">🤖</div>
        <div class="app-body">
          <h2>FinManus 超级智能体</h2>
          <p>ReAct 推理 · 7 类工具 · 自主决策</p>
          <ul>
            <li>开放式工具调用与多步推理</li>
            <li>知识库检索 + 金融计算</li>
            <li>复杂理财任务一站式处理</li>
          </ul>
        </div>
        <button class="app-btn">启动智能体 →</button>
      </div>
    </section>

    <section class="tech-stack">
      <span v-for="t in techStack" :key="t" class="tech-tag">{{ t }}</span>
    </section>

    <AppFooter />
  </div>
</template>

<script setup>
import { useRouter } from 'vue-router'
import { useHead } from '@vueuse/head'
import AppFooter from '../components/AppFooter.vue'

useHead({
  title: 'FinAdvisor - 金融理财智能平台',
  meta: [{ name: 'description', content: '面向金融理财咨询场景的企业级智能平台' }],
})

const router = useRouter()
const go = (path) => router.push(path)

const features = [
  { icon: '🔍', title: 'RAG 知识检索', desc: '百炼向量知识库，理财文档与监管政策精准召回' },
  { icon: '🧠', title: '四层记忆', desc: 'Working / Episodic / Semantic / Summary 分层记忆' },
  { icon: '🛡️', title: 'Agent Harness', desc: '工具安全边界、Checkpoint 恢复、运行工件落盘' },
  { icon: '⚡', title: '流式对话', desc: 'SSE 实时输出，多轮咨询上下文连贯' },
]

const techStack = ['FastAPI', 'LangChain LCEL', 'LangGraph', 'RAG', 'ReAct', 'MCP', 'DashScope']
</script>

<style scoped>
.home {
  min-height: 100vh;
  background: linear-gradient(160deg, #0a1628 0%, #12243d 40%, #1a3a5c 100%);
  color: #e8edf4;
}

.hero {
  text-align: center;
  padding: 72px 24px 48px;
}

.hero-badge {
  display: inline-block;
  font-size: 12px;
  letter-spacing: 2px;
  text-transform: uppercase;
  color: #c9a227;
  border: 1px solid rgba(201, 162, 39, 0.4);
  padding: 4px 14px;
  border-radius: 20px;
  margin-bottom: 20px;
}

.hero-title {
  font-size: 3.5rem;
  font-weight: 700;
  background: linear-gradient(135deg, #ffffff, #c9a227);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
  margin-bottom: 8px;
}

.hero-sub {
  font-size: 1.25rem;
  color: rgba(255, 255, 255, 0.75);
  margin-bottom: 16px;
}

.hero-desc {
  font-size: 0.95rem;
  color: rgba(255, 255, 255, 0.5);
  line-height: 1.8;
}

.features {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 20px;
  max-width: 1000px;
  margin: 0 auto 48px;
  padding: 0 24px;
}

.feature-card {
  background: rgba(255, 255, 255, 0.05);
  border: 1px solid rgba(255, 255, 255, 0.08);
  border-radius: 12px;
  padding: 24px;
  text-align: center;
}

.feature-icon { font-size: 2rem; display: block; margin-bottom: 12px; }
.feature-card h3 { font-size: 1rem; margin-bottom: 8px; color: #fff; }
.feature-card p { font-size: 0.85rem; color: rgba(255,255,255,0.55); line-height: 1.6; }

.apps {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 32px;
  max-width: 1100px;
  margin: 0 auto;
  padding: 0 24px 48px;
}

.app-card {
  width: 100%;
  max-width: 480px;
  background: #fff;
  color: #1a2332;
  border-radius: 16px;
  padding: 32px;
  cursor: pointer;
  transition: transform 0.25s, box-shadow 0.25s;
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.app-card:hover {
  transform: translateY(-6px);
  box-shadow: 0 20px 40px rgba(0, 0, 0, 0.25);
}

.app-card.advisor { border-top: 4px solid #1a7f6e; }
.app-card.manus { border-top: 4px solid #c9a227; }

.app-icon { font-size: 2.5rem; }

.app-body h2 { font-size: 1.4rem; margin-bottom: 8px; }
.app-body p { font-size: 0.9rem; color: #5a6a7e; margin-bottom: 12px; }
.app-body ul {
  list-style: none;
  font-size: 0.85rem;
  color: #5a6a7e;
  line-height: 1.9;
}
.app-body ul li::before { content: '✓ '; color: #1a7f6e; font-weight: bold; }

.app-btn {
  align-self: flex-start;
  background: #0a1628;
  color: #fff;
  border: none;
  padding: 10px 24px;
  border-radius: 8px;
  font-size: 0.95rem;
  cursor: pointer;
  transition: background 0.2s;
}
.app-card.manus .app-btn { background: #8a6d1a; }
.app-btn:hover { opacity: 0.9; }

.tech-stack {
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 10px;
  padding: 0 24px 40px;
}

.tech-tag {
  font-size: 0.8rem;
  padding: 6px 14px;
  border-radius: 20px;
  background: rgba(255,255,255,0.08);
  color: rgba(255,255,255,0.6);
  border: 1px solid rgba(255,255,255,0.1);
}

@media (max-width: 768px) {
  .hero-title { font-size: 2.5rem; }
  .apps { flex-direction: column; align-items: center; }
}
</style>

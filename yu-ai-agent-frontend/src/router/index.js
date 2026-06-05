import { createRouter, createWebHistory } from 'vue-router'

const routes = [
  {
    path: '/',
    name: 'Home',
    component: () => import('../views/Home.vue'),
    meta: {
      title: 'FinAdvisor - 金融理财智能平台',
      description: '面向金融理财咨询场景的企业级智能平台，提供 RAG 知识问答、数值测算与 ReAct 超级智能体',
    },
  },
  {
    path: '/advisor',
    name: 'FinAdvisor',
    component: () => import('../views/FinAdvisor.vue'),
    meta: {
      title: '理财顾问 - FinAdvisor',
      description: '金融理财咨询助手，支持知识问答、复利定投测算、投资报告生成',
    },
  },
  {
    path: '/manus',
    name: 'FinManus',
    component: () => import('../views/FinManus.vue'),
    meta: {
      title: 'FinManus 超级智能体 - FinAdvisor',
      description: 'ReAct 金融超级智能体，自主调用工具完成复杂理财分析任务',
    },
  },
  // 兼容旧路由
  { path: '/love-master', redirect: '/advisor' },
  { path: '/super-agent', redirect: '/manus' },
]

const router = createRouter({
  history: createWebHistory(),
  routes,
})

router.beforeEach((to, _from, next) => {
  if (to.meta.title) document.title = to.meta.title
  next()
})

export default router

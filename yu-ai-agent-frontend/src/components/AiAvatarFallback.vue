<template>
  <div class="avatar" :class="type">
    <span v-if="type === 'advisor'">📊</span>
    <span v-else-if="type === 'manus'">🤖</span>
    <span v-else>💰</span>
  </div>
</template>

<script setup>
defineProps({
  type: { type: String, default: 'advisor' },
})
</script>

<style scoped>
.avatar {
  width: 100%;
  height: 100%;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 18px;
  border-radius: 50%;
}

.advisor {
  background: linear-gradient(135deg, #1a7f6e, #2aab96);
}

.manus {
  background: linear-gradient(135deg, #8a6d1a, #c9a227);
}

.default, .super, .love {
  background: linear-gradient(135deg, #0a1628, #1a3a5c);
}
</style>

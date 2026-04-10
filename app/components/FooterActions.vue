<script setup lang="ts">
const store = useMtgaStore();
const startAllPending = ref(false);

const handleStartAll = async () => {
  if (startAllPending.value) {
    return;
  }
  startAllPending.value = true;
  try {
    await store.runProxyStartAll();
  } finally {
    startAllPending.value = false;
  }
};
</script>

<template>
  <div class="flex flex-wrap items-center justify-between gap-4">
    <div>
      <div class="text-sm font-semibold text-slate-100">快速操作</div>
      <div class="text-xs text-slate-400">一键启动会依次检查网络、证书与 hosts 配置</div>
    </div>
    <MtgaLoadingButton
      class="mtga-btn-primary px-8"
      :loading="startAllPending"
      @click="handleStartAll"
    >
      一键启动全部服务
    </MtgaLoadingButton>
  </div>
</template>

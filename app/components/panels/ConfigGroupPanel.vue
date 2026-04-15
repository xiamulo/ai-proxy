<script setup lang="ts">
import type { ConfigGroup, ProviderId } from "~/composables/mtgaTypes";

const store = useMtgaStore();
const configGroups = store.configGroups;
const currentIndex = store.currentConfigIndex;

const DEFAULT_MIDDLE_ROUTE = "/v1";
const GEMINI_DEFAULT_MIDDLE_ROUTE = "/v1beta";

const editorOpen = ref(false);
const editorMode = ref<"add" | "edit">("add");
const formError = ref("");
const middleRouteEnabled = ref(false);
const availableModels = ref<string[]>([]);
const modelLoading = ref(false);
const formModelDiscoveryStrategy = ref("");
const formModelDiscoveryScope = ref("");

const confirmOpen = ref(false);
const confirmTitle = ref("确认删除");
const confirmMessage = ref("");
const pendingDeleteIndex = ref<number | null>(null);
const pendingSwitchIndex = ref<number | null>(null);
const switchInProgress = ref(false);
const refreshInProgress = ref(false);
const testInProgress = ref(false);
const saveInProgress = ref(false);
const deleteInProgress = ref(false);
const reorderInProgress = ref(false);
const showApiKey = ref(false);

const form = reactive({
  name: "",
  provider: "openai_chat_completion" as ProviderId,
  api_url: "",
  model_id: "",
  api_key: "",
  middle_route: "",
  reasoning_effort: "high" as "none" | "low" | "medium" | "high" | "xhigh",
  prompt_cache_enabled: false,
  request_params_enabled: true,
  websocket_mode_enabled: false,
});

const PROVIDER_LABELS: Record<ProviderId, string> = {
  openai_chat_completion: "OpenAI Chat Completion",
  openai_response: "OpenAI Response",
  anthropic: "Anthropic",
  gemini: "Gemini",
};

const isGpt54OpenAiGroup = (group: Pick<ConfigGroup, "provider" | "model_id">) =>
  ["openai_chat_completion", "openai_response"].includes(normalizeProvider(group.provider)) &&
  (group.model_id || "").trim().toLowerCase() === "gpt-5.4";

const isGpt54OpenAiForm = computed(() =>
  isGpt54OpenAiGroup({
    provider: form.provider,
    model_id: form.model_id,
  }),
);

const normalizeProvider = (provider?: string): ProviderId => {
  if (
    provider === "openai_chat_completion" ||
    provider === "openai_response" ||
    provider === "anthropic" ||
    provider === "gemini"
  ) {
    return provider;
  }
  return "openai_chat_completion";
};

const getProviderLabel = (provider?: string) => PROVIDER_LABELS[normalizeProvider(provider)];

const normalizeApiUrl = (value: string) => value.trim().replace(/\/+$/, "");

const getDefaultMiddleRoute = (provider: ProviderId) =>
  provider === "gemini" ? GEMINI_DEFAULT_MIDDLE_ROUTE : DEFAULT_MIDDLE_ROUTE;

const supportsModelDiscovery = (_provider: ProviderId) => true;

const selectedIndex = computed({
  get: () => (configGroups.value.length ? currentIndex.value : -1),
  set: (value) => {
    if (value < 0 || value >= configGroups.value.length) {
      return;
    }
    if (value === currentIndex.value && pendingSwitchIndex.value === null) {
      return;
    }
    currentIndex.value = value;
    pendingSwitchIndex.value = value;
    showApiKey.value = false;
    void processConfigSwitch();
  },
});

const processConfigSwitch = async () => {
  if (switchInProgress.value) {
    return;
  }
  switchInProgress.value = true;
  try {
    while (pendingSwitchIndex.value !== null) {
      const targetIndex = pendingSwitchIndex.value;
      pendingSwitchIndex.value = null;
      currentIndex.value = targetIndex;

      const saved = await store.saveConfig();
      if (!saved) {
        store.appendLog("保存配置组失败");
        continue;
      }

      if (pendingSwitchIndex.value !== null) {
        continue;
      }
      await store.runProxyApplyCurrentConfig();
    }
  } finally {
    switchInProgress.value = false;
  }
};

const hasSelection = computed(
  () =>
    configGroups.value.length > 0 &&
    selectedIndex.value >= 0 &&
    selectedIndex.value < configGroups.value.length,
);

const panelActionBusy = computed(
  () =>
    switchInProgress.value ||
    saveInProgress.value ||
    deleteInProgress.value ||
    reorderInProgress.value,
);

const normalizeMiddleRoute = (value: string, provider: ProviderId = form.provider) => {
  let raw = value.trim();
  if (!raw) {
    raw = getDefaultMiddleRoute(provider);
  }
  if (!raw.startsWith("/")) {
    raw = `/${raw}`;
  }
  if (raw.length > 1) {
    raw = raw.replace(/\/+$/, "");
    if (!raw) {
      raw = "/";
    }
  }
  return raw;
};

const buildModelDiscoveryScope = (payload: {
  provider?: string;
  api_url: string;
  api_key?: string;
  middle_route?: string;
}) => {
  const provider = normalizeProvider(payload.provider);
  return JSON.stringify([
    provider,
    normalizeApiUrl(payload.api_url),
    (payload.api_key || "").trim(),
    normalizeMiddleRoute(payload.middle_route || "", provider),
  ]);
};

const setFormModelDiscoveryState = (
  strategyId: string | null | undefined,
  payload?: {
    provider?: string;
    api_url: string;
    api_key?: string;
    middle_route?: string;
  },
) => {
  const normalizedStrategyId = (strategyId || "").trim();
  if (!normalizedStrategyId || !payload) {
    formModelDiscoveryStrategy.value = "";
    formModelDiscoveryScope.value = "";
    return;
  }
  formModelDiscoveryStrategy.value = normalizedStrategyId;
  formModelDiscoveryScope.value = buildModelDiscoveryScope(payload);
};

const isProviderDefaultMiddleRoute = (value: string, provider: ProviderId) =>
  normalizeMiddleRoute(value, provider) === getDefaultMiddleRoute(provider);

watch(
  () => form.provider,
  (provider, previousProvider) => {
    if (!middleRouteEnabled.value || !previousProvider) {
      return;
    }

    const rawMiddleRoute = form.middle_route.trim();
    if (!rawMiddleRoute) {
      form.middle_route = getDefaultMiddleRoute(provider);
      return;
    }

    if (isProviderDefaultMiddleRoute(rawMiddleRoute, previousProvider)) {
      form.middle_route = getDefaultMiddleRoute(provider);
    }
  },
);

const getDisplayName = (group: ConfigGroup, index: number) =>
  group.name?.trim() || `配置组 ${index + 1}`;

const refreshList = async () => {
  if (refreshInProgress.value) {
    return;
  }
  refreshInProgress.value = true;
  try {
    const ok = await store.loadConfig();
    if (ok) {
      store.appendLog("已刷新配置组列表");
    }
  } finally {
    refreshInProgress.value = false;
  }
};

const requestTest = async () => {
  if (testInProgress.value) {
    return;
  }
  if (!hasSelection.value) {
    store.appendLog("请先选择要测活的配置组");
    return;
  }
  testInProgress.value = true;
  try {
    await store.runConfigGroupTest(selectedIndex.value);
  } finally {
    testInProgress.value = false;
  }
};

const resetForm = () => {
  form.name = "";
  form.provider = "openai_chat_completion";
  form.api_url = "";
  form.model_id = "";
  form.api_key = "";
  form.middle_route = "";
  form.reasoning_effort = "high";
  form.prompt_cache_enabled = false;
  form.request_params_enabled = true;
  form.websocket_mode_enabled = false;
  middleRouteEnabled.value = false;
  formError.value = "";
  availableModels.value = [];
  modelLoading.value = false;
  setFormModelDiscoveryState(undefined);
};

const openAdd = () => {
  editorMode.value = "add";
  resetForm();
  editorOpen.value = true;
};

const openEdit = () => {
  if (!hasSelection.value) {
    store.appendLog("请先选择要修改的配置组");
    return;
  }
  editorMode.value = "edit";
  const group = configGroups.value[selectedIndex.value];
  if (!group) {
    return;
  }
  form.name = group.name || "";
  form.provider = normalizeProvider(group.provider);
  form.api_url = group.api_url || "";
  form.model_id = group.model_id || "";
  form.api_key = group.api_key || "";
  form.middle_route = group.middle_route || "";
  form.reasoning_effort = group.reasoning_effort || "high";
  form.prompt_cache_enabled = group.prompt_cache_enabled ?? false;
  form.request_params_enabled = group.request_params_enabled ?? true;
  form.websocket_mode_enabled = group.websocket_mode_enabled ?? false;
  middleRouteEnabled.value = Boolean(group.middle_route);
  formError.value = "";
  availableModels.value = [];
  setFormModelDiscoveryState(group.model_discovery_strategy, {
    provider: group.provider,
    api_url: group.api_url || "",
    api_key: group.api_key || "",
    middle_route: group.middle_route || "",
  });
  editorOpen.value = true;
};

const closeEditor = () => {
  editorOpen.value = false;
};

const hasDuplicateConfigGroup = (payload: ConfigGroup, ignoredIndex: number | null = null) =>
  configGroups.value.some((group, index) => {
    if (ignoredIndex !== null && index === ignoredIndex) {
      return false;
    }
    return (
      normalizeProvider(group.provider) === normalizeProvider(payload.provider) &&
      normalizeApiUrl(group.api_url || "") === normalizeApiUrl(payload.api_url) &&
      (group.model_id || "").trim() === payload.model_id &&
      (group.api_key || "").trim() === payload.api_key &&
      normalizeMiddleRoute(group.middle_route || "", normalizeProvider(group.provider)) ===
        normalizeMiddleRoute(payload.middle_route || "", normalizeProvider(payload.provider))
    );
  });

const handleSave = async () => {
  if (saveInProgress.value) {
    return;
  }
  const payload: ConfigGroup = {
    name: form.name.trim(),
    provider: form.provider,
    api_url: normalizeApiUrl(form.api_url),
    model_id: form.model_id.trim(),
    api_key: form.api_key.trim(),
    reasoning_effort: form.reasoning_effort,
    prompt_cache_enabled: form.prompt_cache_enabled,
    request_params_enabled: form.request_params_enabled,
    websocket_mode_enabled: isGpt54OpenAiForm.value ? form.websocket_mode_enabled : false,
  };

  if (!payload.api_url || !payload.model_id || !payload.api_key) {
    formError.value = "API URL、实际模型ID 和 API Key 都是必填项";
    store.appendLog("错误: API URL、实际模型ID和API Key都是必填项");
    return;
  }

  if (middleRouteEnabled.value && form.middle_route.trim()) {
    payload.middle_route = normalizeMiddleRoute(form.middle_route, form.provider);
  } else {
    delete payload.middle_route;
  }

  if (
    formModelDiscoveryStrategy.value &&
    formModelDiscoveryScope.value === buildModelDiscoveryScope(payload)
  ) {
    payload.model_discovery_strategy = formModelDiscoveryStrategy.value;
  } else {
    delete payload.model_discovery_strategy;
  }

  const editingIndex =
    editorMode.value === "edit" && hasSelection.value ? selectedIndex.value : null;
  if (hasDuplicateConfigGroup(payload, editingIndex)) {
    formError.value = "相同 provider、API URL、实际模型ID、API Key 和中间路由的配置组已存在";
    store.appendLog("错误: 相同 provider、API URL、实际模型ID、API Key 和中间路由的配置组已存在");
    return;
  }

  if (editorMode.value === "add") {
    configGroups.value.push(payload);
    currentIndex.value = configGroups.value.length - 1;
  } else if (hasSelection.value) {
    configGroups.value.splice(selectedIndex.value, 1, payload);
  }

  saveInProgress.value = true;
  try {
    const ok = await store.saveConfig();
    if (ok) {
      const displayName = getDisplayName(payload, selectedIndex.value);
      store.appendLog(
        editorMode.value === "add"
          ? `已添加配置组: ${displayName}`
          : `已修改配置组: ${displayName}`,
      );
      closeEditor();
    } else {
      store.appendLog("保存配置组失败");
    }
  } finally {
    saveInProgress.value = false;
  }
};

const handleFetchModels = async () => {
  if (modelLoading.value) {
    return;
  }
  const apiUrl = form.api_url.trim();
  if (!apiUrl) {
    store.appendLog("获取模型列表失败: API URL为空");
    return;
  }
  if (!supportsModelDiscovery(form.provider)) {
    store.appendLog("当前提供商不支持通过 /models 自动发现模型，请直接手填实际模型ID");
    return;
  }
  modelLoading.value = true;
  const requestPayload = {
    provider: form.provider,
    api_url: apiUrl,
    api_key: form.api_key.trim(),
    model_id: form.model_id.trim(),
    middle_route: middleRouteEnabled.value
      ? normalizeMiddleRoute(form.middle_route, form.provider)
      : "",
  };
  const result = await store.fetchConfigGroupModels(requestPayload);
  if (result !== null) {
    availableModels.value = result.models;
    setFormModelDiscoveryState(result.strategyId, requestPayload);
  }
  modelLoading.value = false;
};

const requestDelete = () => {
  if (!hasSelection.value) {
    store.appendLog("请先选择要删除的配置组");
    return;
  }
  if (configGroups.value.length <= 1) {
    store.appendLog("至少需要保留一个配置组");
    return;
  }
  const group = configGroups.value[selectedIndex.value];
  if (!group) {
    return;
  }
  pendingDeleteIndex.value = selectedIndex.value;
  confirmTitle.value = "确认删除";
  confirmMessage.value = `确定要删除配置组 “${getDisplayName(group, selectedIndex.value)}” 吗？`;
  confirmOpen.value = true;
};

const cancelDelete = () => {
  confirmOpen.value = false;
  pendingDeleteIndex.value = null;
};

const confirmDelete = async () => {
  if (deleteInProgress.value) {
    return;
  }
  if (pendingDeleteIndex.value == null) {
    return;
  }
  deleteInProgress.value = true;
  try {
    const index = pendingDeleteIndex.value;
    const group = configGroups.value[index];
    if (!group) {
      store.appendLog("配置组不存在，已取消删除");
      confirmOpen.value = false;
      pendingDeleteIndex.value = null;
      return;
    }
    configGroups.value.splice(index, 1);
    if (currentIndex.value >= configGroups.value.length) {
      currentIndex.value = Math.max(configGroups.value.length - 1, 0);
    } else if (currentIndex.value > index) {
      currentIndex.value -= 1;
    }
    const ok = await store.saveConfig();
    if (ok) {
      store.appendLog(`已删除配置组: ${getDisplayName(group, index)}`);
    } else {
      store.appendLog("保存配置组失败");
    }
    confirmOpen.value = false;
    pendingDeleteIndex.value = null;
  } finally {
    deleteInProgress.value = false;
  }
};

const moveUp = async () => {
  if (reorderInProgress.value) {
    return;
  }
  if (!hasSelection.value || selectedIndex.value <= 0) {
    return;
  }
  const index = selectedIndex.value;
  const current = configGroups.value[index];
  const prev = configGroups.value[index - 1];
  if (!current || !prev) {
    return;
  }
  reorderInProgress.value = true;
  try {
    configGroups.value[index - 1] = current;
    configGroups.value[index] = prev;
    currentIndex.value = index - 1;
    await store.saveConfig();
  } finally {
    reorderInProgress.value = false;
  }
};

const moveDown = async () => {
  if (reorderInProgress.value) {
    return;
  }
  if (!hasSelection.value || selectedIndex.value >= configGroups.value.length - 1) {
    return;
  }
  const index = selectedIndex.value;
  const current = configGroups.value[index];
  const next = configGroups.value[index + 1];
  if (!current || !next) {
    return;
  }
  reorderInProgress.value = true;
  try {
    configGroups.value[index + 1] = current;
    configGroups.value[index] = next;
    currentIndex.value = index + 1;
    await store.saveConfig();
  } finally {
    reorderInProgress.value = false;
  }
};
</script>

<template>
  <div class="space-y-5">
    <section class="mtga-card space-y-4">
      <div class="flex flex-col gap-3 md:flex-row md:items-center md:justify-between p-5 pb-0">
        <div>
          <h2 class="text-lg font-semibold text-mtga-text">配置组管理</h2>
          <p class="text-sm text-mtga-text-muted">支持不同接口的多配置无缝切换</p>
        </div>
        <div class="flex flex-wrap items-center gap-2">
          <button class="mtga-btn-ghost" :disabled="panelActionBusy" @click="refreshList">
            刷新
          </button>
          <button class="mtga-btn-ghost" :disabled="panelActionBusy" @click="requestTest">
            测活
          </button>
          <div class="w-px h-5 bg-slate-700/60 mx-1"></div>
          <button class="mtga-btn-ghost" :disabled="panelActionBusy" @click="openEdit">修改</button>
          <button
            class="mtga-btn-ghost text-red-400! hover:text-red-300! hover:bg-red-500/10!"
            :disabled="panelActionBusy"
            @click="requestDelete"
          >
            删除
          </button>
          <button class="mtga-btn-primary" :disabled="panelActionBusy" @click="openAdd">
            新增配置组
          </button>
        </div>
      </div>

      <div class="grid gap-4 lg:grid-cols-[280px,1fr]">
        <div class="rounded-2xl border border-mtga-border bg-slate-900/50 p-3">
          <div class="mb-2 flex items-center justify-between text-xs text-mtga-text-muted">
            <span>配置组列表</span>
            <div class="flex items-center gap-2">
              <button
                class="mtga-btn-icon"
                :disabled="panelActionBusy || !hasSelection || selectedIndex <= 0"
                @click="moveUp"
              >
                ↑
              </button>
              <button
                class="mtga-btn-icon"
                :disabled="
                  panelActionBusy || !hasSelection || selectedIndex >= configGroups.length - 1
                "
                @click="moveDown"
              >
                ↓
              </button>
            </div>
          </div>
          <div
            v-if="!configGroups.length"
            class="rounded-xl border border-dashed border-mtga-border px-4 py-8 text-center"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              class="mx-auto h-8 w-8 text-slate-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="1.5"
                d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10"
              />
            </svg>
            <p class="mt-2 text-sm text-slate-500">暂无配置组</p>
            <button class="mtga-btn-primary btn-xs! mt-3" @click="openAdd">新增配置组</button>
          </div>
          <div class="mt-3 space-y-2">
            <button
              v-for="(group, index) in configGroups"
              :key="index"
              type="button"
              class="w-full rounded-xl border p-3 text-left transition-all active:scale-[0.98]"
              :class="[
                index === selectedIndex
                  ? 'border-cyan-500 bg-cyan-900/30 shadow-[0_0_12px_rgba(6,182,212,0.15)] ring-1 ring-cyan-500/30'
                  : 'border-mtga-border bg-slate-800/50 hover:border-cyan-500/40 hover:bg-slate-800',
              ]"
              @click="selectedIndex = index"
            >
              <div class="flex items-center justify-between gap-3">
                <div class="min-w-0">
                  <div class="truncate font-medium text-slate-200">
                    {{ getDisplayName(group, index) }}
                  </div>
                  <div class="mt-1 truncate text-xs text-slate-500">
                    {{ getProviderLabel(group.provider) }}
                  </div>
                </div>
                <span v-if="index === currentIndex" class="mtga-chip">当前</span>
              </div>
            </button>
          </div>
        </div>

        <div class="rounded-2xl border border-mtga-border bg-slate-900/50 p-5">
          <template v-if="hasSelection">
            <div class="flex flex-wrap items-start justify-between gap-3">
              <div>
                <h3 class="text-base font-semibold text-slate-100">
                  {{ getDisplayName(configGroups[selectedIndex]!, selectedIndex) }}
                </h3>
                <p class="text-sm text-mtga-text-muted">
                  {{ getProviderLabel(configGroups[selectedIndex]?.provider) }}
                </p>
              </div>
              <div class="flex flex-wrap gap-2">
                <span class="mtga-chip">{{ configGroups[selectedIndex]?.model_id }}</span>
                <span v-if="isGpt54OpenAiGroup(configGroups[selectedIndex]!)" class="mtga-chip">
                  {{ `思考:${configGroups[selectedIndex]?.reasoning_effort || "high"}` }}
                </span>
                <span
                  v-if="configGroups[selectedIndex]?.request_params_enabled ?? true"
                  class="mtga-chip"
                  >透传参数</span
                >
                <span v-else class="mtga-chip">精简参数</span>
                <span v-if="configGroups[selectedIndex]?.websocket_mode_enabled" class="mtga-chip"
                  >WebSocket</span
                >
              </div>
            </div>

            <div class="mt-4 space-y-2.5">
              <div class="flex items-baseline gap-3">
                <span class="shrink-0 text-xs font-medium text-slate-500 w-16">API URL</span>
                <span class="break-all text-sm text-slate-300">{{
                  configGroups[selectedIndex]?.api_url
                }}</span>
              </div>
              <div class="flex items-center gap-3">
                <span class="shrink-0 text-xs font-medium text-slate-500 w-16">API Key</span>
                <span class="break-all text-sm font-mono text-slate-300">{{
                  showApiKey
                    ? configGroups[selectedIndex]?.api_key || "未设置"
                    : configGroups[selectedIndex]?.api_key
                      ? configGroups[selectedIndex]!.api_key.slice(0, 3) +
                        "****" +
                        configGroups[selectedIndex]!.api_key.slice(-4)
                      : "未设置"
                }}</span>
                <button
                  v-if="configGroups[selectedIndex]?.api_key"
                  class="mtga-btn-icon"
                  @click="showApiKey = !showApiKey"
                >
                  {{ showApiKey ? "隐藏" : "显示" }}
                </button>
              </div>
              <div class="flex items-baseline gap-3">
                <span class="shrink-0 text-xs font-medium text-slate-500 w-16">中间路由</span>
                <span class="break-all text-sm text-slate-300">{{
                  configGroups[selectedIndex]?.middle_route ||
                  getDefaultMiddleRoute(normalizeProvider(configGroups[selectedIndex]?.provider))
                }}</span>
              </div>
              <div class="flex items-baseline gap-3">
                <span class="shrink-0 text-xs font-medium text-slate-500 w-16">思考强度</span>
                <span class="text-sm text-slate-300">{{
                  configGroups[selectedIndex]?.reasoning_effort || "high"
                }}</span>
              </div>
              <div class="flex items-baseline gap-3">
                <span class="shrink-0 text-xs font-medium text-slate-500 w-16">提示缓存2</span>
                <span class="text-sm text-slate-300">{{
                  configGroups[selectedIndex]?.prompt_cache_enabled ? "已启用" : "未启用"
                }}</span>
              </div>
            </div>

            <div
              class="mt-5 rounded-2xl bg-slate-800/80 px-4 py-3 text-xs leading-6 text-mtga-text-muted"
            >
              <div>测活会发送最小对话请求来验证配置组。</div>
              <div>
                关闭“带上请求参数”后，运行代理与测活都会尽量去掉 temperature、top_p 等额外参数。
              </div>
              <div v-if="configGroups[selectedIndex]?.websocket_mode_enabled">
                当前配置会在 `OpenAI + gpt-5.4` 的流式请求里优先尝试上游 WebSocket 模式。
              </div>
            </div>
          </template>
          <div
            v-else
            class="rounded-2xl border border-dashed border-mtga-border px-4 py-10 text-center"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              class="mx-auto h-10 w-10 text-slate-600"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                stroke-linecap="round"
                stroke-linejoin="round"
                stroke-width="1.5"
                d="M15 15l-2 5L9 9l11 4-5 2zm0 0l5 5M7.188 2.239l.777 2.897M5.136 7.965l-2.898-.777M13.95 4.05l-2.122 2.122m-5.657 5.656l-2.12 2.122"
              />
            </svg>
            <p class="mt-3 text-sm text-slate-500">请选择一个配置组查看详情</p>
          </div>
        </div>
      </div>
    </section>
  </div>

  <ConfigGroupEditorDialog
    v-model:open="editorOpen"
    v-model:name="form.name"
    v-model:provider="form.provider"
    v-model:api-url="form.api_url"
    v-model:model-id="form.model_id"
    v-model:api-key="form.api_key"
    v-model:middle-route="form.middle_route"
    v-model:reasoning-effort="form.reasoning_effort"
    v-model:middle-route-enabled="middleRouteEnabled"
    v-model:prompt-cache-enabled="form.prompt_cache_enabled"
    v-model:request-params-enabled="form.request_params_enabled"
    v-model:websocket-mode-enabled="form.websocket_mode_enabled"
    :mode="editorMode"
    :form-error="formError"
    :default-middle-route="getDefaultMiddleRoute(form.provider)"
    :available-models="availableModels"
    :model-loading="modelLoading"
    :saving="saveInProgress"
    @fetch-models="handleFetchModels"
    @save="handleSave"
    @cancel="closeEditor"
  />

  <ConfirmDialog
    :open="confirmOpen"
    :title="confirmTitle"
    :message="confirmMessage"
    :loading="deleteInProgress"
    confirm-text="删除"
    @confirm="confirmDelete"
    @cancel="cancelDelete"
  />
</template>

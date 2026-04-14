<script setup lang="ts">
import type { ProviderId } from "~/composables/mtgaTypes";

const props = withDefaults(
  defineProps<{
    open?: boolean;
    mode?: "add" | "edit";
    name?: string;
    provider?: ProviderId;
    apiUrl?: string;
    modelId?: string;
    apiKey?: string;
    middleRoute?: string;
    middleRouteEnabled?: boolean;
    promptCacheEnabled?: boolean;
    requestParamsEnabled?: boolean;
    websocketModeEnabled?: boolean;
    formError?: string;
    defaultMiddleRoute?: string;
    availableModels?: string[];
    modelLoading?: boolean;
    saving?: boolean;
  }>(),
  {
    open: false,
    mode: "add",
    name: "",
    provider: "openai_chat_completion",
    apiUrl: "",
    modelId: "",
    apiKey: "",
    middleRoute: "",
    middleRouteEnabled: false,
    promptCacheEnabled: false,
    requestParamsEnabled: true,
    websocketModeEnabled: true,
    formError: "",
    defaultMiddleRoute: "/v1",
    availableModels: () => [],
    modelLoading: false,
    saving: false,
  },
);

const emit = defineEmits<{
  (event: "update:open", value: boolean): void;
  (event: "update:name", value: string): void;
  (event: "update:provider", value: ProviderId): void;
  (event: "update:apiUrl", value: string): void;
  (event: "update:modelId", value: string): void;
  (event: "update:apiKey", value: string): void;
  (event: "update:middleRoute", value: string): void;
  (event: "update:middleRouteEnabled", value: boolean): void;
  (event: "update:promptCacheEnabled", value: boolean): void;
  (event: "update:requestParamsEnabled", value: boolean): void;
  (event: "update:websocketModeEnabled", value: boolean): void;
  (event: "save"): void;
  (event: "cancel"): void;
  (event: "fetch-models"): void;
}>();

const openModel = computed({
  get: () => props.open,
  set: (value: boolean) => emit("update:open", value),
});

const nameModel = computed({
  get: () => props.name,
  set: (value: string) => emit("update:name", value),
});

const providerModel = computed({
  get: () => props.provider,
  set: (value: ProviderId) => emit("update:provider", value),
});

const apiUrlModel = computed({
  get: () => props.apiUrl,
  set: (value: string) => emit("update:apiUrl", value),
});

const modelIdModel = computed({
  get: () => props.modelId,
  set: (value: string) => emit("update:modelId", value),
});

const apiKeyModel = computed({
  get: () => props.apiKey,
  set: (value: string) => emit("update:apiKey", value),
});

const middleRouteModel = computed({
  get: () => props.middleRoute,
  set: (value: string) => emit("update:middleRoute", value),
});

const middleRouteEnabledModel = computed({
  get: () => props.middleRouteEnabled,
  set: (value: boolean) => emit("update:middleRouteEnabled", value),
});

const promptCacheEnabledModel = computed({
  get: () => props.promptCacheEnabled,
  set: (value: boolean) => emit("update:promptCacheEnabled", value),
});

const requestParamsEnabledModel = computed({
  get: () => props.requestParamsEnabled,
  set: (value: boolean) => emit("update:requestParamsEnabled", value),
});

const websocketModeEnabledModel = computed({
  get: () => props.websocketModeEnabled,
  set: (value: boolean) => emit("update:websocketModeEnabled", value),
});

const showWebsocketModeOption = computed(
  () =>
    (props.provider === "openai_chat_completion" || props.provider === "openai_response") &&
    props.modelId.trim().toLowerCase() === "gpt-5.4",
);

const handleDialogClose = () => {
  emit("cancel");
};

const handleCancel = () => {
  openModel.value = false;
  emit("cancel");
};

const handleSave = () => {
  if (props.saving) {
    return;
  }
  emit("save");
};

const providerOptions: { label: string; value: ProviderId }[] = [
  { label: "OpenAI Chat Completion", value: "openai_chat_completion" },
  { label: "OpenAI Response", value: "openai_response" },
  { label: "Anthropic", value: "anthropic" },
  { label: "Gemini", value: "gemini" },
];
</script>

<template>
  <MtgaDialog v-model:open="openModel" max-width="max-w-l" @close="handleDialogClose">
    <template #header>
      <div class="flex items-center justify-between gap-3">
        <div>
          <h3 class="text-lg font-semibold text-slate-100">
            {{ props.mode === "add" ? "新增配置组" : "修改配置组" }}
          </h3>
          <p class="text-xs text-slate-400">配置代理目标与鉴权参数</p>
        </div>
        <span class="mtga-chip">配置编辑</span>
      </div>
    </template>

    <div class="px-6 py-6 space-y-5">
      <MtgaInput
        v-model="nameModel"
        label="配置组名称"
        placeholder="例如：我的常用配置"
        icon="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
      />

      <MtgaSelect v-model="providerModel" label="接口规范" :options="providerOptions" />

      <!-- API 接口地址与代理路由 -->
      <div class="space-y-2 rounded-2xl border border-slate-700 bg-slate-900/50 px-4 py-3">
        <label class="flex cursor-pointer items-center gap-2">
          <input
            v-model="middleRouteEnabledModel"
            type="checkbox"
            class="toggle toggle-xs toggle-info"
          />
          <span class="label-text text-sm font-medium text-slate-300">自定义接口代理路由</span>
        </label>
        <div class="text-[11px] leading-5 text-slate-500">
          默认情况下将采用选定“接口规范”的内置路由。若上游提供了完整代理地址可在此覆盖修改。<br />
          若填写空值或斜杠，则会代理到上游域名根目录。
        </div>

        <template v-if="middleRouteEnabledModel">
          <div class="divider my-2 border-slate-700"></div>
          <MtgaInput
            v-model="apiUrlModel"
            label="API 接口根地址"
            placeholder="例如：https://api.openai.com"
            required
            :error="props.formError"
          />
          <MtgaInput
            v-model="middleRouteModel"
            label="覆盖中间代理路由"
            placeholder="将覆盖默认的对话、模型列举等代理前缀"
          />
        </template>
        <template v-else>
          <div class="divider my-2 border-slate-700"></div>
          <MtgaInput
            v-model="apiUrlModel"
            label="API 接口地址"
            placeholder="例如：https://api.openai.com"
            required
            :error="props.formError"
          />
        </template>
      </div>

      <!-- API Key 授权 -->
      <MtgaInput
        v-model="apiKeyModel"
        label="API Key"
        placeholder="输入用于请求验证的 Bearer Token"
        required
        :error="props.formError"
      />

      <!-- 模型 ID 与自动发现 -->
      <div class="space-y-2 rounded-2xl border border-slate-700 bg-slate-900/50 px-4 py-3">
        <label class="flex cursor-pointer items-center gap-2">
          <input
            v-model="promptCacheEnabledModel"
            type="checkbox"
            class="toggle toggle-xs toggle-info"
          />
          <span class="label-text text-sm font-medium text-slate-300">提示缓存</span>
        </label>
        <div class="text-[11px] leading-5 text-slate-500">
          当客户端请求列出所有可用模型时，代理应当如何处理。
        </div>

        <div class="divider my-2 border-slate-700"></div>

        <MtgaInput
          v-model="modelIdModel"
          label="主要对话模型"
          placeholder="例如：gpt-4o，在通过代理列举可用模型时将包含此项"
          required
          :error="props.formError"
        />

        <label class="flex cursor-pointer items-center gap-2">
          <input
            v-model="requestParamsEnabledModel"
            type="checkbox"
            class="toggle toggle-xs toggle-info"
          />
          <span class="label-text text-sm font-medium text-slate-300">带上请求参数</span>
        </label>
        <p class="text-xs leading-5 text-slate-400">
          默认保持和之前一致，会透传 temperature、top_p
          等请求参数；关闭后仅发送基础消息字段，适合旧模型或兼容性较差的上游。
        </p>

        <template v-if="showWebsocketModeOption">
          <div class="divider my-2 border-slate-700"></div>

          <label class="flex cursor-pointer items-center gap-2">
            <input
              v-model="websocketModeEnabledModel"
              type="checkbox"
              class="toggle toggle-xs toggle-info"
            />
            <span class="label-text text-sm font-medium text-slate-300">启用 WebSocket 模式</span>
          </label>
          <p class="text-xs leading-5 text-slate-400">
            仅对 `OpenAI + gpt-5.4` 生效。开启后，流式请求会优先尝试使用 OpenAI Responses WebSocket
            模式；连接失败时会自动回退到普通 HTTP。
          </p>
        </template>
      </div>
      <div
        v-if="props.formError"
        class="alert alert-error py-2 px-3 rounded-xl border border-red-500/30 bg-red-900/20 text-red-400"
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          class="stroke-current shrink-0 h-4 w-4"
          fill="none"
          viewBox="0 0 24 24"
        >
          <path
            stroke-linecap="round"
            stroke-linejoin="round"
            stroke-width="2"
            d="M10 14l2-2m0 0l2-2m-2 2l-2-2m2 2l2 2m7-2a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <span class="text-xs">{{ props.formError }}</span>
      </div>
    </div>

    <template #footer>
      <button class="mtga-btn-dialog-ghost flex-1" @click="handleCancel">取消</button>
      <button
        class="mtga-btn-dialog-primary flex-1"
        :class="props.saving ? 'loading' : ''"
        :disabled="props.saving"
        @click="handleSave"
      >
        保存
      </button>
    </template>
  </MtgaDialog>
</template>

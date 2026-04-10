<template>
  <div class="h-screen flex flex-col overflow-hidden">
    <!-- 统一的毛玻璃容器，现在充满整个窗口 -->
    <div
      class="flex-1 flex flex-col bg-slate-950/70 backdrop-blur-xl overflow-hidden shadow-inner shadow-cyan-500/5"
    >
      <!-- 顶部标题区域 -->
      <header class="w-full px-8 py-3 shrink-0 border-b border-slate-800/60">
        <div class="flex flex-wrap items-center justify-between gap-4">
          <div class="flex items-center gap-3">
            <h1 class="text-lg font-semibold tracking-tight text-white">
              <slot name="title">AI Proxy</slot>
            </h1>
            <span class="text-xs text-slate-500">
              <slot name="subtitle">一站式 AI 代理服务器管理面板</slot>
            </span>
          </div>
        </div>
      </header>

      <!-- 中间内容区域 -->
      <main class="flex-1 flex min-h-0">
        <!-- 左侧区域 (占据 7/12) -->
        <section class="flex-7 flex flex-col min-w-0">
          <slot name="left" />
        </section>

        <!-- 垂直分割线 -->
        <div class="w-px bg-slate-800/60 self-stretch shrink-0"></div>

        <!-- 右侧区域 (占据 5/12) -->
        <section class="flex-5 flex flex-col min-w-0">
          <slot name="right" />
        </section>
      </main>

      <!-- 底部操作区域 -->
      <footer class="p-5 shrink-0 border-t border-slate-800/60">
        <slot name="footer" />
      </footer>
    </div>
  </div>
</template>

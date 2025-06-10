import { defineConfig } from 'vitepress'
// import markdownItAnchor from 'markdown-it-anchor'

// https://vitepress.dev/reference/site-config
export default defineConfig({
  title: "yorhaha's blog",
  description: "yorhaha's blog",
  themeConfig: {
    // https://vitepress.dev/reference/default-theme-config
    nav: [
      { text: 'Home', link: '/' },
    ],
    search: {
      provider: 'local'
    },
    sidebar: [
      {
        text: 'Development',
        items: [
          { text: 'Prompt', link: '/prompt' },
          { text: 'Linux', link: '/linux' },
          { text: 'LLM Inference', link: '/llm_inference' },
        ]
      },
      {
        text: 'KVCache',
        items: [
          { text: 'ClusterKV', link: '/cluster-kv' },
          { text: 'MoBA', link: '/moba' },
          { text: 'KIVI', link: '/kivi' },
          { text: 'Tensor Product Attention', link: '/tpa' },
          { text: 'CAKE', link: '/cake' },
          { text: 'OmniKV', link: '/omini-kv' },
          { text: 'Ada-KV', link: '/ada-kv' },
        ]
      },
      {
        text: 'LLM',
        items: [
          { text: 'Qwen3', link: '/qwen3' },
          { text: 'DeepSeek V2', link: '/deepseek-v2' },
          { text: 'RoPE', link: '/rope' },
        ]
      },
      {
        text: 'Machine Learning',
        items: [
          { text: 'AUC', link: '/auc' },
          { text: 'SiLU', link: '/silu' },
        ]
      },
      {
        text: 'LLM4REC',
        items: [
          { text: 'NoteLLM', link: '/note_llm' },
          { text: 'InfoNCE', link: '/info_nce' },
          { text: '[doing] GenIR survey', link: '/genir_survey' },
        ]
      }
    ],

    socialLinks: [
      { icon: 'github', link: 'https://github.com/yorhaha' }
    ]
  },
  markdown: {
    container: {
      tipLabel: '提示',
      warningLabel: '警告',
      dangerLabel: '危险',
      infoLabel: '信息',
      detailsLabel: '详细信息'
    },
    lineNumbers: true,
    image: {
      lazyLoading: true,
    },
    math: true,
    // anchor: {
    //   permalink: markdownItAnchor.permalink.headerLink()
    // },
    toc: { level: [0, 1, 2] },
    anchor: {
      level: [1, 2, 3],
    }
  },
  lastUpdated: true,
})

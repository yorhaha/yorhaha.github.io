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
        text: 'KVCache',
        items: [
          { text: 'ClusterKV', link: '/cluster-kv' },
        ]
      },
      {
        text: 'LLM',
        items: [
          { text: 'Qwen3', link: '/qwen3' },
          { text: 'DeepSeek V2', link: '/deepseek-v2' },
        ]
      },
      {
        text: 'Machine Learning',
        items: [
          { text: 'AUC', link: '/auc' },
          { text: 'SiLU', link: '/silu' },
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

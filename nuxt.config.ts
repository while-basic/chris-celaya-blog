// https://nuxt.com/docs/api/configuration/nuxt-config
import { defineNuxtConfig } from 'nuxt/config'

export default defineNuxtConfig({
  // https://github.com/nuxt-themes/alpine
  extends: '@nuxt-themes/alpine',

  modules: [
    // https://github.com/nuxt-modules/plausible
    '@nuxtjs/plausible',
    // https://github.com/nuxt/devtools
    '@nuxt/devtools'
  ],

  app: {
    head: {
      script: [
        {
          src: '/_vercel/insights/script.js',
          defer: true
        }
      ]
    }
  },

  content: {
    highlight: {
      theme: {
        default: 'github-light',
        dark: 'github-dark'
      },
      preload: [
        'python',
        'javascript',
        'typescript',
        'bash',
        'shell',
        'json',
        'markdown'
      ]
    }
  }
})

// https://github.com/nuxt-themes/alpine/blob/main/nuxt.schema.ts
export default defineAppConfig({
  alpine: {
    title: 'Blog',
    description: 'A minimalist blog theme',
    image: {
      src: '/social-card-preview.png',
      alt: 'An image showcasing my project.',
      width: 400,
      height: 300
    },
    header: {
      position: 'right', // possible value are : | 'left' | 'center' | 'right'
      logo: {
        path: '/logo.svg', // path of the logo
        pathDark: '/logo-dark.svg', // path of the logo in dark mode, leave this empty if you want to use the same logo
        alt: 'alpine' // alt of the logo
      }
    },
    footer: {
      credits: {
        enabled: false, // text at bottom of footer, possible value are : true | false
        repository: 'https://www.github.com/while-basic/chris-celaya-blog' // our github repository
      },
      navigation: true, // possible value are : true | false
      alignment: 'center', // possible value are : 'none' | 'left' | 'center' | 'right'
      message: 'Follow me on' // string that will be displayed in the footer (leave empty or delete to disable)
    },
    socials: {
      twitter: 'Im_Mr_Chris',
      instagram: 'chriscelaya',
      linkedin: {
        icon: 'uil:linkedin',
        label: 'LinkedIn',
        href: 'https://www.linkedin.com/in/christophercelaya'
      }
    },
    prose: {
      copyButton: {
        iconCopy: 'ph:copy',
        iconCopied: 'ph:check'
      },
      headings: {
        icon: 'ph:link'
      },
      code: {
        theme: {
          default: 'github-light',
          dark: 'github-dark'
        },
        languages: {
          python: 'Python',
          js: 'JavaScript',
          ts: 'TypeScript',
          bash: 'Bash',
          shell: 'Shell'
        },
        highlighter: 'shiki',
        options: {
          defaultLanguage: 'plaintext',
          background: true
        }
      }
    }
  }
})

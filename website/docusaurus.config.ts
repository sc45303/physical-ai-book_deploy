import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

const config: Config = {
  title: 'Physical AI & Humanoid Robotics Course',
  tagline: 'Bridging digital AI brains and physical humanoid bodies',
  favicon: 'img/icon.png',

  // Move future flags into customFields (optional)
  customFields: {
    future: {
      experimental_faster: {
        ssgWorkerThreads: true,
      },
      v4: {
        removeLegacyPostBuildHeadAttribute: true,
      },
    },
  },

  plugins: [
    async function chatbotPlugin(context, options) {
      return {
        name: 'chatbot-plugin',
        clientModules: [
          require.resolve('./src/components/ChatbotInjector.jsx'),
        ],
      };
    },
  ],

  url: 'https://physicalai-course.com',
  baseUrl: '/',

  organizationName: 'physicalai-course',
  projectName: 'physical-ai-humanoid-course',

  onBrokenLinks: 'warn',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          editUrl:
            'https://github.com/physicalai-course/physical-ai-humanoid-course/edit/main/website/',
        },
        blog: {
          showReadingTime: true,
          feedOptions: { type: ['rss', 'atom'], xslt: true },
          editUrl:
            'https://github.com/physicalai-course/physical-ai-humanoid-course/edit/main/website/',
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/docusaurus-social-card.jpg',
    colorMode: { respectPrefersColorScheme: true },
    navbar: {
      title: 'Physical AI Course',
      logo: { alt: 'Physical AI & Humanoid Robotics Course Logo', src: 'img/logo.png' },
      items: [
        { type: 'docSidebar', sidebarId: 'tutorialSidebar', position: 'left', label: 'Modules' },
        { href: 'https://github.com/physicalai-course/physical-ai-humanoid-course', label: 'GitHub', position: 'right' },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Modules',
          items: [
            { label: 'Module 1: ROS 2', to: '/modules/ros2/intro' },
            { label: 'Module 2: Simulation', to: '/modules/simulation/intro' },
            { label: 'Module 3: Isaac', to: '/modules/isaac/intro' },
            { label: 'Module 4: VLA', to: '/modules/vla/intro' },
          ],
        },
        {
          title: 'Community',
          items: [
            { label: 'Discord', href: 'https://discordapp.com/invite/physicalai' },
            { label: 'X', href: 'https://x.com/physicalai' },
          ],
        },
        {
          title: 'More',
          items: [{ label: 'GitHub', href: 'https://github.com/physicalai-course/physical-ai-humanoid-course' }],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} Physical AI & Humanoid Robotics Course. Built with Docusaurus.`,
    },
    prism: { theme: prismThemes.github, darkTheme: prismThemes.dracula },
  } satisfies Preset.ThemeConfig,
};

export default config;

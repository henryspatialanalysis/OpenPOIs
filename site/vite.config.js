import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import sitemap from 'vite-plugin-sitemap'

export default defineConfig({
  base: '/',
  plugins: [
    vue(),
    sitemap({
      hostname: 'https://openpois.org',
      // index.html, about.html and taxonomy.html are globbed automatically from
      // the build output. Only /docs/ (the Sphinx site) must be listed here: it
      // is copied into dist/docs AFTER `vite build` by the deploy workflow, so
      // the plugin's glob can't see it at sitemap-generation time.
      dynamicRoutes: ['/docs/'],
      // Keep the hand-written public/robots.txt (content signals + AI opt-out).
      // The plugin would otherwise overwrite it with a generated stub.
      generateRobotsTxt: false,
    }),
  ],
  build: {
    rollupOptions: {
      output: {
        manualChunks(id) {
          if (id.includes('node_modules/ol-mapbox-style')) return 'ol-mapbox-style'
          if (id.includes('node_modules/ol-pmtiles')) return 'ol-pmtiles'
          if (id.includes('node_modules/ol/')) return 'ol'
          if (id.includes('node_modules/vue')) return 'vue'
        },
      },
    },
  },
})

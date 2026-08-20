import { defineConfig } from 'astro/config';
import preact from '@astrojs/preact';
import mdx from '@astrojs/mdx';

// GitHub Pages project site for SharathSPhD/game-llm.
// Served at https://SharathSPhD.github.io/game-llm — hence the base path.
export default defineConfig({
  site: 'https://SharathSPhD.github.io',
  base: '/game-llm',
  trailingSlash: 'ignore',
  integrations: [preact({ compat: true }), mdx()],
});

const createNextIntlPlugin = require('next-intl/plugin');

const withNextIntl = createNextIntlPlugin('./i18n/request.ts');

/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Include .data/*.json in serverless function bundles so fs.readFile
  // can access seed data (labs, activities, etc.) at runtime on Vercel.
  outputFileTracingIncludes: {
    '/api/**': ['./.data/**/*.json'],
  },
  async rewrites() {
    return [
      // Proxy compute network API to Workers backend
      {
        source: '/api/compute/:path*',
        destination: 'https://labfork-agents.jonathan-hawkins.workers.dev/api/compute/:path*',
      },
      // Proxy contributor API to compute leaderboard
      {
        source: '/api/contributor/leaderboard',
        destination: 'https://labfork-agents.jonathan-hawkins.workers.dev/api/compute/leaderboard',
      },
      {
        source: '/api/contributor/:userId',
        destination: 'https://labfork-agents.jonathan-hawkins.workers.dev/api/compute/devices/:userId/stats',
      },
      // Proxy orchestration API to Workers backend (for legacy routes)
      {
        source: '/api/orchestrator/:path*',
        destination: 'https://labfork-agents.jonathan-hawkins.workers.dev/api/:path*',
      },
    ];
  },
}

module.exports = withNextIntl(nextConfig);

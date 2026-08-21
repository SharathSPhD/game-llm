/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  // Explicitly disable static generation for dynamic routes
  experimental: {},
};

module.exports = nextConfig;

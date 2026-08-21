#!/usr/bin/env node
/**
 * Smoke test: Start next server and curl each page for 200 status.
 * Usage: npm run check-pages
 */

import { spawn } from "child_process";
import { setTimeout as sleep } from "timers/promises";

const baseUrl = "http://localhost:3000";
const pages = ["/", "/lab", "/qre", "/auction", "/studio", "/findings"];
const startupDelay = 6000; // 6 seconds to let server start

async function checkPage(path) {
  try {
    const response = await fetch(`${baseUrl}${path}`, {
      method: "GET",
      headers: { "Accept": "text/html" },
    });
    const status = response.status;
    const ok = status === 200;
    console.log(`  ${ok ? "✓" : "✗"} ${path.padEnd(15)} ${status}`);
    return ok;
  } catch (err) {
    console.log(`  ✗ ${path.padEnd(15)} error: ${err.message}`);
    return false;
  }
}

async function main() {
  console.log("Starting Next.js server...");
  const server = spawn("npm", ["start"], {
    cwd: process.cwd(),
    stdio: "inherit",
  });

  // Wait for server to start
  await sleep(startupDelay);

  console.log(`\nChecking pages (${pages.length} routes):\n`);

  const results = await Promise.all(pages.map((p) => checkPage(p)));
  const passed = results.filter(Boolean).length;
  const total = results.length;

  console.log(`\nResult: ${passed}/${total} routes returned 200`);

  // Kill server
  server.kill();
  process.exit(passed === total ? 0 : 1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});

// Kinetic AI gateway — unified front door via Cloudflare Worker.
// Architecture:
//   API paths (/health, /api/*)  -> GB10 FastAPI backend (URL in KV)
//   Everything else              -> Vercel app (web UI)
// Public URL: https://kinetic.sharath-sathish.workers.dev (stable, never changes)

const VERCEL = "https://kinetic-web.vercel.app";
const API = ["/health", "/api/"];
const CORS = {
  "access-control-allow-origin": "*",
  "access-control-allow-methods": "GET,POST,OPTIONS",
  "access-control-allow-headers": "content-type,authorization",
};

export default {
  async fetch(request, env) {
    const url = new URL(request.url);
    const isApi = API.some((p) => url.pathname === p || url.pathname.startsWith(p));

    if (request.method === "OPTIONS" && isApi) {
      return new Response(null, { headers: CORS });
    }

    if (isApi) {
      const backend = await env.KV.get("gateway_url");
      if (!backend) {
        return json({ error: "gateway not configured" }, 503);
      }

      const target = backend.replace(/\/$/, "") + url.pathname + url.search;
      let resp;
      try {
        resp = await fetch(target, {
          method: request.method,
          headers: { "content-type": request.headers.get("content-type") || "application/json" },
          body: (request.method === "GET" || request.method === "HEAD") ? undefined : await request.arrayBuffer(),
        });
      } catch (e) {
        return json({ error: "backend unreachable", detail: String(e) }, 502);
      }

      const h = new Headers(resp.headers);
      for (const [k, v] of Object.entries(CORS)) h.set(k, v);
      return new Response(await resp.arrayBuffer(), { status: resp.status, headers: h });
    }

    // Proxy Vercel app (UI + its own /api/* routes). Relative asset paths
    // keep working; Host is set from the target URL by fetch.
    const target = VERCEL + url.pathname + url.search;
    const fwd = new Headers(request.headers);
    fwd.delete("host");

    const resp = await fetch(target, {
      method: request.method,
      headers: fwd,
      body: (request.method === "GET" || request.method === "HEAD") ? undefined : await request.arrayBuffer(),
      redirect: "manual",
    });

    return new Response(resp.body, { status: resp.status, headers: resp.headers });
  },
};

function json(o, s) {
  return new Response(JSON.stringify(o), {
    status: s,
    headers: { "content-type": "application/json", ...CORS },
  });
}

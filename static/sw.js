const STATIC_CACHE = "atlas-shell-v2";
const RUNTIME_CACHE = "atlas-runtime-v2";

const PRECACHE_URLS = [
  "/app",
  "/manifest.json",
  "/api/modules"
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => cache.addAll(PRECACHE_URLS)).catch(() => {})
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys
          .filter((k) => k !== STATIC_CACHE && k !== RUNTIME_CACHE)
          .map((k) => caches.delete(k))
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (req.method !== "GET") return;

  const url = new URL(req.url);

  // Кэширование просмотренных тайлов карты OpenStreetMap и скриптов Leaflet для работы в роуминге/офлайне
  if (url.hostname.includes("tile.openstreetmap.org") || url.hostname.includes("unpkg.com")) {
    event.respondWith(
      caches.open(RUNTIME_CACHE).then(async (cache) => {
        const cached = await cache.match(req);
        if (cached) return cached;
        try {
          const fresh = await fetch(req);
          if (fresh && fresh.status === 200) {
            cache.put(req, fresh.clone());
          }
          return fresh;
        } catch (_) {
          return cached || Response.error();
        }
      })
    );
    return;
  }

  // Только одноимённый origin для страниц и GET API
  if (url.origin !== self.location.origin) return;

  // Network-first с возвратом сохранённой копии при отсутствии интернета
  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.status === 200) {
          const copy = res.clone();
          const targetCache = (url.pathname === "/" || url.pathname === "/app" || url.pathname === "/manifest.json")
            ? STATIC_CACHE
            : RUNTIME_CACHE;
          caches.open(targetCache).then((cache) => cache.put(req, copy)).catch(() => {});
        }
        return res;
      })
      .catch(async () => {
        const cached = await caches.match(req);
        if (cached) return cached;
        if (req.mode === "navigate") {
          return (await caches.match("/app")) || Response.error();
        }
        return Response.error();
      })
  );
});

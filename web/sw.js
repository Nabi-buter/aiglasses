// Offline support: once the page has been opened, it keeps working without a network
// (useful in exam halls). Large model/runtime files are cache-first; small app files
// and head.json are network-first so a redeploy is picked up on the next load.
const CACHE = "ai-glasses-v1";
const BIG = /\/(models\/.+\.(task|tflite)|vendor\/.+\.(wasm|mjs)|vendor\/wasm\/.+\.js)$/;
const PRECACHE = [
  "./", "index.html", "styles.css", "app.js", "pipeline.js", "manifest.webmanifest",
  "vendor/vision_bundle.mjs", "vendor/wasm/vision_wasm_internal.js", "vendor/wasm/vision_wasm_internal.wasm",
  "models/face_landmarker.task", "models/mobilenet_v3_large.tflite", "icon.svg",
];

self.addEventListener("install", (e) => {
  e.waitUntil(caches.open(CACHE).then((c) => c.addAll(PRECACHE)).then(() => self.skipWaiting()));
});
self.addEventListener("activate", (e) => {
  e.waitUntil(caches.keys()
    .then((keys) => Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))))
    .then(() => self.clients.claim()));
});
self.addEventListener("fetch", (e) => {
  const url = new URL(e.request.url);
  if (e.request.method !== "GET" || url.origin !== location.origin) return;
  if (BIG.test(url.pathname)) {
    e.respondWith(caches.match(e.request).then((hit) => hit || fetch(e.request).then((res) => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); }
      return res;
    })));
    return;
  }
  e.respondWith(fetch(e.request).then((res) => {
    if (res.ok) { const copy = res.clone(); caches.open(CACHE).then((c) => c.put(e.request, copy)); }
    return res;
  }).catch(() => caches.match(e.request, { ignoreSearch: true })));
});

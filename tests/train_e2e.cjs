// End-to-end test of the in-browser training page.
// node tests/train_e2e.cjs <baseUrl> <dataset.zip> [screenshot.png]
// Uploads the ZIP, trains, applies the model on this device, and checks the live page uses it.
// Prints the trained head.json (JSON) on stdout.
const { chromium } = require("playwright");

(async () => {
  const [base, zip, shot] = process.argv.slice(2);
  const browser = await chromium.launch({ args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"] });
  const ctx = await browser.newContext({ viewport: { width: 1100, height: 900 }, deviceScaleFactor: shot ? 2 : 1, permissions: ["camera"] });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  await page.route("**/models/head.json", (r) => r.fulfill({ status: 404, body: "" }));  // no bundled model
  await page.goto(`${base}/web/train.html`);
  await page.setInputFiles("#zipIn", zip);
  // 1st run without the shipped base data, so the result can be compared with train.py on the same photos.
  const hasBase = await page.waitForFunction(() => window.__train.base !== undefined, null, { timeout: 60000 }).then(() => page.evaluate(() => !!window.__train.base));
  if (hasBase) await page.uncheck("#withBase");
  await page.waitForFunction(() => !document.getElementById("trainBtn").disabled, null, { timeout: 30000 });
  const t0 = Date.now();
  await page.click("#trainBtn");
  await page.waitForFunction(() => window.__train.head || /못했|중지/.test(document.getElementById("pStage").textContent),
    null, { timeout: 900000, polling: 500 });
  const stage = await page.textContent("#pStage");
  const head = await page.evaluate(() => window.__train.head);
  if (!head) throw new Error("training failed: " + stage + " / " + (await page.textContent("#pDetail")));
  const secs = (Date.now() - t0) / 1000;
  await page.waitForTimeout(800);   // thumbnails

  // 2nd run with the base data merged in (the default for users).
  let merged = null;
  if (hasBase) {
    await page.check("#withBase");
    await page.evaluate(() => (window.__train.head = null));
    await page.click("#trainBtn");
    await page.waitForFunction(() => window.__train.head || /못했|중지/.test(document.getElementById("pStage").textContent),
      null, { timeout: 1800000, polling: 1000 });
    const h2 = await page.evaluate(() => window.__train.head);
    if (!h2?.n_base || !h2.cv?.user) throw new Error("merged training failed: " + (await page.textContent("#pDetail")));
    merged = { n_base: h2.n_base, n_user: h2.n_user, cv: h2.cv };
    await page.waitForTimeout(500);
  }
  if (shot) await page.screenshot({ path: shot, fullPage: true });

  // Use it on this device, then the live page must pick it up.
  await page.click("#useHere");
  await page.goto(`${base}/web/index.html`);
  await page.waitForFunction(() => window.__aig?.head, null, { timeout: 60000 });
  const live = await page.evaluate(() => ({ source: window.__aig.headSource, trained_at: window.__aig.head.trained_at, svc: document.getElementById("svcText").textContent }));
  const applied = await page.evaluate(() => JSON.parse(localStorage.getItem("aiGlassesCustomHead.v1") || "null")?.trained_at);
  if (live.source !== "custom" || live.trained_at !== applied) throw new Error("live page did not load the trained model: " + JSON.stringify(live));
  const bad = errors.filter((e) => !e.startsWith("INFO:") && !/head\.json|404/.test(e));
  if (bad.length) throw new Error("page errors: " + bad.join(" | "));
  process.stdout.write(JSON.stringify({ head, secs, live, merged }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });

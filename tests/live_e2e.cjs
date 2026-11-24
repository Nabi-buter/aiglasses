// End-to-end test of the live view with a fake webcam.
// node tests/live_e2e.cjs <baseUrl> <head.json path | "none"> <cam.y4m> [screenshot.png]
// With "none" there is no trained head: only the built-in glasses/no-glasses model runs.
// Prints JSON: the tracked faces (left to right) with verdicts, and checks on the UI.
const fs = require("fs");
const { chromium } = require("playwright");

(async () => {
  const [base, headPath, y4m, shot] = process.argv.slice(2);
  const browser = await chromium.launch({
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-video-capture=${y4m}`],
  });
  const ctx = await browser.newContext({ viewport: { width: 1360, height: 900 }, deviceScaleFactor: shot ? 2 : 1, permissions: ["camera"] });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
  const noHead = headPath === "none";
  await page.route("**/models/head.json", (r) => (noHead ? r.fulfill({ status: 404, body: "" }) : r.fulfill({ path: headPath, contentType: "application/json" })));
  await page.goto(`${base}/web/index.html`);

  // Wait until both people have a stable, averaged verdict.
  await page.waitForFunction((noHead) => {
    const s = window.__aig;
    const t = s?.tracker?.tracks?.filter((x) => x.visible) || [];
    return (noHead ? s.presence && !s.head : s.head) && t.length >= 2 && t.every((x) => x.stable && x.samples.length >= 4);
  }, noHead, { timeout: 90000, polling: 250 });
  await page.waitForTimeout(600);

  const result = await page.evaluate(() => {
    const s = window.__aig;
    const tracks = s.tracker.tracks.filter((x) => x.visible).sort((a, b) => a.box.x0 - b.box.x0);
    return {
      tracks: tracks.map((t) => ({ id: t.id, verdict: t.verdict, probs: t.probs, samples: t.samples.length, facePx: Math.round(t.facePx) })),
      ids: s.tracker.nextId - 1,
      log: s.log.map((l) => ({ verdict: l.verdict, pAi: l.pAi, src: l.src })),
      people: [...document.querySelectorAll("#people .person")].map((b) => b.textContent),
      noModelTag: document.getElementById("noModelTag").hidden ? null : document.getElementById("noModelTag").textContent,
      perf: s.perf,
    };
  });

  // Select the red one and screenshot the page (detail panel visible).
  const red = result.tracks.find((t) => t.verdict === "AI_GLASSES" || t.verdict === "SUSPECTED");
  if (red) await page.evaluate((id) => { window.__aig.selectedId = id; window.__aig.dirty = true; }, red.id);
  await page.waitForTimeout(500);

  // Collect a few training frames and export the ZIP.
  await page.evaluate(() => (document.getElementById("collectBox").open = true));
  await page.click("#cOne"); await page.click("#cOne");
  await page.waitForFunction(() => window.__aig.collect.items.length >= 2);
  const [dl] = await Promise.all([page.waitForEvent("download"), page.click("#cZip")]);
  const zipPath = (shot || "/tmp/aiglasses-e2e.png").replace(/\.png$/, "_collected.zip");
  await dl.saveAs(zipPath);
  result.zip = { name: dl.suggestedFilename(), bytes: fs.statSync(zipPath).size, path: zipPath };

  if (shot) {
    await page.evaluate(() => (document.getElementById("collectBox").open = false));
    await page.waitForTimeout(300);
    await page.screenshot({ path: shot, fullPage: true });
    await page.locator("#stage").screenshot({ path: shot.replace(/\.png$/, "_stage.png") });
  }
  result.errors = errors;
  process.stdout.write(JSON.stringify(result));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });

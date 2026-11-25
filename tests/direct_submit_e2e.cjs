// "바로 적용" on the training page, against a MOCK GitHub API (nothing is written to GitHub).
// node tests/direct_submit_e2e.cjs <baseUrl> <dataset.zip> <out-contribution.json>
// Checks: a bad token is refused; a good token is stored; the contribution is committed as one
// Git commit on main (blob -> tree -> commit -> ref), retrying when main moved meanwhile;
// the page notices the redeployed shared model. Writes the committed file for the Python merge test.
const fs = require("fs");
const { chromium } = require("playwright");

(async () => {
  const [base, zip, out] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));

  const calls = [];
  let blob = null, refPatches = 0, deployed = false;
  const realHead = JSON.parse(fs.readFileSync("web/models/head.json", "utf8"));
  await page.route("**/models/head.json*", (r) => r.fulfill({
    contentType: "application/json",
    body: JSON.stringify(deployed ? { ...realHead, trained_at: "2099-01-01T00:00:00+00:00", n_contributions: 1 } : realHead),
  }));
  await page.route("https://api.github.com/**", async (r) => {
    const req = r.request(), url = new URL(req.url()), auth = req.headers()["authorization"] || "";
    const p = url.pathname, m = req.method();
    calls.push(`${m} ${p}`);
    const json = (status, body) => r.fulfill({ status, contentType: "application/json", body: JSON.stringify(body) });
    if (!auth.endsWith(" good-token")) return json(401, { message: "Bad credentials" });
    if (m === "GET" && /^\/repos\/[^/]+\/[^/]+$/.test(p)) return json(200, { permissions: { push: true } });
    if (m === "GET" && p.endsWith("/git/ref/heads/main")) return json(200, { object: { sha: refPatches ? "c_moved" : "c0" } });
    if (m === "GET" && /\/git\/commits\//.test(p)) return json(200, { tree: { sha: "t0" } });
    if (m === "POST" && p.endsWith("/git/blobs")) { blob = JSON.parse(req.postData()); return json(201, { sha: "b1" }); }
    if (m === "POST" && p.endsWith("/git/trees")) {
      const t = JSON.parse(req.postData());
      if (!/^contributions\/c_.+\.json$/.test(t.tree[0].path)) return json(422, { message: "bad path " + t.tree[0].path });
      return json(201, { sha: "t1" });
    }
    if (m === "POST" && p.endsWith("/git/commits")) return json(201, { sha: "c1", html_url: "https://github.com/x/y/commit/c1" });
    if (m === "PATCH" && p.endsWith("/git/refs/heads/main")) {
      refPatches++;
      if (refPatches === 1) return json(422, { message: "Update is not a fast forward" });   // main moved meanwhile
      deployed = true;
      return json(200, { object: { sha: "c1" } });
    }
    return json(404, { message: "not mocked: " + m + " " + p });
  });

  await page.goto(`${base}/web/train.html`);
  await page.waitForFunction(() => window.__train.base !== undefined, null, { timeout: 60000 });
  await page.evaluate(() => (window.__train.pollMs = 300));
  await page.setInputFiles("#zipIn", zip);
  await page.waitForFunction(() => window.__train.items.size > 0);
  await page.check("#consent");

  // bad token -> refused, nothing stored
  await page.fill("#tokenIn", "bad-token");
  await page.click("#tokenSave");
  await page.waitForFunction(() => /토큰/.test(document.getElementById("shareMsg").textContent));
  const badMsg = await page.textContent("#shareMsg");
  if (await page.evaluate(() => localStorage.getItem("aiGlassesGithubToken.v1"))) throw new Error("bad token was stored");

  // good token -> stored, direct submit enabled
  await page.fill("#tokenIn", "good-token");
  await page.click("#tokenSave");
  await page.waitForSelector("#submitDirect:not([disabled])", { timeout: 30000 });
  await page.click("#submitDirect");
  await page.waitForFunction(() => /업데이트되었습니다|못했|오류|시간 초과/.test(document.getElementById("pStage").textContent + document.getElementById("shareMsg").textContent),
    null, { timeout: 900000, polling: 500 });
  const stage = await page.textContent("#pStage"), msg = await page.textContent("#shareMsg");
  if (!/업데이트되었습니다/.test(stage)) throw new Error(`direct submit failed: ${stage} / ${msg}`);
  if (!blob) throw new Error("no blob was sent");
  const file = Buffer.from(blob.content, "base64").toString("utf8");
  fs.writeFileSync(out, file);
  const c = JSON.parse(file);
  process.stdout.write(JSON.stringify({ badMsg: badMsg.trim().slice(0, 120), refPatches, calls, id: c.id, n: c.n, counts: c.counts, stage, errors }));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });

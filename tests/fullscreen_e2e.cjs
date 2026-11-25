// Full-screen camera view + training-data counts.
// node tests/fullscreen_e2e.cjs <baseUrl> <cam.y4m> [screenshot-prefix]
// Desktop and phone (touch): the button enters full screen, the frame fills the screen, no buttons are
// visible, a swipe UP does nothing, a swipe DOWN returns to the normal page. The training page shows the
// number of photos per class used by the shared model.
const { chromium, devices } = require("playwright");

async function check(browser, base, ctxOpts, touch, shot) {
  const ctx = await browser.newContext({ ...ctxOpts, permissions: ["camera"] });
  const page = await ctx.newPage();
  const errors = [];
  page.on("pageerror", (e) => errors.push(e.message));
  await page.goto(`${base}/web/index.html`);
  await page.waitForFunction(() => window.__aig?.stream && document.getElementById("view").width > 0, null, { timeout: 60000 });
  await page.click("#fsBtn");
  await page.waitForFunction(() => window.__aig.fs && document.getElementById("stage").classList.contains("fs"));
  await page.waitForTimeout(700);
  const vp = page.viewportSize();
  const inFs = await page.evaluate(() => {
    const c = document.getElementById("view").getBoundingClientRect(), st = document.getElementById("stage").getBoundingClientRect();
    const visibleButtons = [...document.querySelectorAll("button, .btn, input")].filter((b) => {
      const r = b.getBoundingClientRect();
      if (!r.width || !r.height) return false;
      const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      return top && (top === b || b.contains(top));
    }).map((b) => b.id || b.textContent.trim());
    return { canvas: [c.width, c.height], stage: [st.width, st.height], visibleButtons,
      hint: getComputedStyle(document.getElementById("fsHint")).opacity, native: !!document.fullscreenElement };
  });
  if (shot) await page.screenshot({ path: `${shot}_${touch ? "phone" : "desktop"}.png` });
  const fills = Math.abs(inFs.stage[0] - vp.width) < 2 && Math.abs(inFs.stage[1] - vp.height) < 2 &&
    (Math.abs(inFs.canvas[0] - vp.width) < 2 || Math.abs(inFs.canvas[1] - vp.height) < 2);
  if (!fills) throw new Error("full screen does not fill the viewport: " + JSON.stringify({ vp, inFs }));
  if (inFs.visibleButtons.length) throw new Error("buttons visible in full screen: " + inFs.visibleButtons.join(", "));

  const swipe = async (dy) => {
    const x = vp.width / 2, y0 = dy > 0 ? vp.height * 0.25 : vp.height * 0.75;
    if (touch) {
      const cdp = await ctx.newCDPSession(page);
      const pt = (y) => [{ x, y }];
      await cdp.send("Input.dispatchTouchEvent", { type: "touchStart", touchPoints: pt(y0) });
      for (let i = 1; i <= 6; i++) await cdp.send("Input.dispatchTouchEvent", { type: "touchMove", touchPoints: pt(y0 + (dy * i) / 6) });
      await cdp.send("Input.dispatchTouchEvent", { type: "touchEnd", touchPoints: [] });
    } else {
      await page.mouse.move(x, y0); await page.mouse.down();
      await page.mouse.move(x, y0 + dy, { steps: 8 }); await page.mouse.up();
    }
    await page.waitForTimeout(400);
  };
  await swipe(-vp.height * 0.4);                       // up: must stay in full screen
  const afterUp = await page.evaluate(() => window.__aig.fs);
  await swipe(vp.height * 0.4);                        // down: leave
  const afterDown = await page.evaluate(() => ({ fs: window.__aig.fs, cls: document.getElementById("stage").classList.contains("fs"), native: !!document.fullscreenElement }));
  if (!afterUp) throw new Error("swipe up left full screen");
  if (afterDown.fs || afterDown.cls || afterDown.native) throw new Error("swipe down did not leave full screen: " + JSON.stringify(afterDown));
  // keyboard on desktop
  if (!touch) {
    await page.keyboard.press("f"); await page.waitForFunction(() => window.__aig.fs);
    await page.keyboard.press("Escape"); await page.waitForFunction(() => !window.__aig.fs);
  }
  await ctx.close();
  return { viewport: vp, canvas: inFs.canvas, native: inFs.native, errors };
}

(async () => {
  const [base, y4m, shot] = process.argv.slice(2);
  const browser = await chromium.launch({
    args: ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream", `--use-file-for-fake-video-capture=${y4m}`],
  });
  const desktop = await check(browser, base, { viewport: { width: 1280, height: 800 } }, false, shot);
  const phone = await check(browser, base, { ...devices["Pixel 7"] }, true, shot);

  // training page: counts per class for the shared model
  const page = await browser.newPage({ viewport: { width: 1100, height: 900 } });
  await page.goto(`${base}/web/train.html`);
  await page.waitForSelector("#stats table", { timeout: 30000 });
  const stats = await page.evaluate(() => [...document.querySelectorAll("#stats table tr")].map((tr) => [...tr.children].map((c) => c.textContent.trim())));
  if (shot) await page.locator("#statsPanel").screenshot({ path: `${shot}_stats.png` });
  await browser.close();
  process.stdout.write(JSON.stringify({ desktop, phone, stats }));
})().catch((e) => { console.error(e); process.exit(1); });

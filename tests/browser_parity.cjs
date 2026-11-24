// node tests/browser_parity.cjs <baseUrl> <headUrl> <file urls...>  -> JSON on stdout
const { chromium } = require("playwright");
(async () => {
  const [base, head, ...files] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.error("pageerror:", e.message));
  await page.goto(`${base}/tests/parity.html`);
  await page.waitForFunction(() => window.ready === true, null, { timeout: 30000 });
  const out = await page.evaluate(([f, h]) => window.runParity(f, h), [files, head]);
  process.stdout.write(JSON.stringify(out));
  await browser.close();
})().catch((e) => { console.error(e); process.exit(1); });

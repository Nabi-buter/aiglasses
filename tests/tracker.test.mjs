// node --test tests/tracker.test.mjs
import test from "node:test";
import assert from "node:assert/strict";
import { Tracker, iou, meanProbs } from "../web/tracker.js";
import { makeZip, crc32 } from "../web/zip.js";

const det = (x, y, w = 100, h = 50, facePx = 120) => ({
  boxes: { band: { x0: x, y0: y, x1: x + w, y1: y + h }, hinge_l: {}, hinge_r: {} }, facePx, rect: { x0: x, y0: y, x1: x + w, y1: y + h * 3 },
});

test("iou", () => {
  assert.equal(iou({ x0: 0, y0: 0, x1: 10, y1: 10 }, { x0: 0, y0: 0, x1: 10, y1: 10 }), 1);
  assert.equal(iou({ x0: 0, y0: 0, x1: 10, y1: 10 }, { x0: 20, y0: 20, x1: 30, y1: 30 }), 0);
});

test("keeps identities while faces move, two people", () => {
  const tr = new Tracker();
  let ts = tr.update([det(0, 0), det(400, 0)], 0);
  const [a, b] = ts.map((t) => t.id);
  for (let i = 1; i <= 20; i++) ts = tr.update([det(400 + i * 3, i), det(i * 3, i)], i * 33);
  const left = ts.find((t) => t.raw.x0 < 200), right = ts.find((t) => t.raw.x0 > 200);
  assert.equal(left.id, a);
  assert.equal(right.id, b);
  assert.equal(ts.length, 2);
});

test("drops faces that left and gives new faces new ids", () => {
  const tr = new Tracker({ dropAfterMs: 500 });
  tr.update([det(0, 0)], 0);
  let ts = tr.update([], 300);
  assert.equal(ts.length, 1); assert.equal(ts[0].visible, false);
  ts = tr.update([], 900);
  assert.equal(ts.length, 0);
  ts = tr.update([det(0, 0)], 1000);
  assert.equal(ts[0].id, 2);
});

test("averages classifier output and becomes stable after N samples", () => {
  const tr = new Tracker({ stableAfter: 3, minIntervalMs: 100 });
  const [t] = tr.update([det(0, 0)], 0);
  tr.addSample(t, { ai_glasses: 0.9, regular_glasses: 0.1 }, 0);
  assert.equal(t.stable, false);
  tr.update([det(0, 0)], 50);
  assert.equal(tr.next(50), null, "respects min interval");
  tr.update([det(0, 0)], 120);
  assert.equal(tr.next(120), t);
  tr.addSample(t, { ai_glasses: 0.3, regular_glasses: 0.7 }, 120);
  tr.addSample(t, { ai_glasses: 0.6, regular_glasses: 0.4 }, 240);
  assert.equal(t.stable, true);
  assert.ok(Math.abs(t.probs.ai_glasses - 0.6) < 1e-9);
});

test("next() prefers never-classified faces and honours eligibility", () => {
  const tr = new Tracker({ minIntervalMs: 0 });
  const ts = tr.update([det(0, 0, 100, 50, 200), det(400, 0, 100, 50, 30)], 0);
  tr.addSample(ts[0], { a: 1 }, 0);
  assert.equal(tr.next(10), ts[1]);
  assert.equal(tr.next(10, (t) => t.facePx > 50), ts[0]);
});

test("meanProbs", () => {
  assert.deepEqual(meanProbs([{ a: 1, b: 0 }, { a: 0, b: 1 }]), { a: 0.5, b: 0.5 });
  assert.equal(meanProbs([]), null);
});

test("zip has valid structure", async () => {
  assert.equal(crc32(new TextEncoder().encode("123456789")), 0xcbf43926);
  const blob = makeZip([{ name: "ai_glasses/p01__a.jpg", data: new Uint8Array([1, 2, 3]) }, { name: "labels.csv", data: new TextEncoder().encode("x") }]);
  const buf = new DataView(await blob.arrayBuffer());
  assert.equal(buf.getUint32(0, true), 0x04034b50);
  const eocd = buf.byteLength - 22;
  assert.equal(buf.getUint32(eocd, true), 0x06054b50);
  assert.equal(buf.getUint16(eocd + 10, true), 2);
});

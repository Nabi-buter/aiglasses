// Keeps an identity for each face across live frames, smooths the boxes for display,
// and averages the classifier output over several frames, so one bad frame
// (motion blur, a reflection) does not flip a person between red and green.
// Pure logic, no DOM: tests/tracker.test.mjs runs it in Node.

export const TRACK_DEFAULTS = {
  matchIou: 0.15,        // minimum IoU to treat a detection as the same face
  matchDist: 0.6,        // ...or centre distance below this × box width
  dropAfterMs: 1200,     // forget a face not seen for this long
  boxAlpha: 0.55,        // display box smoothing (1 = no smoothing)
  window: 10,            // classifier results averaged per face
  maxAgeMs: 6000,        // ...but only those from the last 6 s
  minIntervalMs: 250,    // do not re-classify the same face more often than this
  stableAfter: 3,        // samples needed before a verdict counts as stable
};

const area = (b) => Math.max(0, b.x1 - b.x0) * Math.max(0, b.y1 - b.y0);
export function iou(a, b) {
  const x0 = Math.max(a.x0, b.x0), y0 = Math.max(a.y0, b.y0);
  const x1 = Math.min(a.x1, b.x1), y1 = Math.min(a.y1, b.y1);
  const inter = Math.max(0, x1 - x0) * Math.max(0, y1 - y0);
  const u = area(a) + area(b) - inter;
  return u > 0 ? inter / u : 0;
}
const centre = (b) => [(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2];

export class Tracker {
  constructor(opts = {}) {
    this.o = { ...TRACK_DEFAULTS, ...opts };
    this.tracks = [];
    this.nextId = 1;
  }

  reset() { this.tracks = []; }

  /**
   * detections: [{boxes:{band,hinge_l,hinge_r}, facePx, rect, ...}] for this frame.
   * Returns the live tracks (seen this frame first).
   */
  update(detections, now) {
    const o = this.o;
    const pairs = [];
    detections.forEach((d, di) => this.tracks.forEach((t, ti) => {
      const s = iou(d.boxes.band, t.raw);
      const [dx, dy] = centre(d.boxes.band), [tx, ty] = centre(t.raw);
      const w = Math.max(1, t.raw.x1 - t.raw.x0);
      const dist = Math.hypot(dx - tx, dy - ty) / w;
      if (s >= o.matchIou || dist < o.matchDist) pairs.push({ di, ti, score: s - dist * 0.1 });
    }));
    pairs.sort((a, b) => b.score - a.score);
    const usedD = new Set(), usedT = new Set();
    for (const p of pairs) {
      if (usedD.has(p.di) || usedT.has(p.ti)) continue;
      usedD.add(p.di); usedT.add(p.ti);
      this._refresh(this.tracks[p.ti], detections[p.di], now);
    }
    detections.forEach((d, di) => {
      if (usedD.has(di)) return;
      const t = { id: this.nextId++, firstSeen: now, samples: [], lastClassified: -Infinity, maxAi: 0, flagged: false };
      this._refresh(t, d, now, true);
      this.tracks.push(t);
    });
    for (const t of this.tracks) t.visible = t.lastSeen === now;
    this.tracks = this.tracks.filter((t) => now - t.lastSeen <= o.dropAfterMs);
    this.tracks.sort((a, b) => (b.visible - a.visible) || a.id - b.id);
    return this.tracks;
  }

  _refresh(t, d, now, fresh = false) {
    const a = fresh ? 1 : this.o.boxAlpha, b = d.boxes.band;
    t.raw = { ...b };
    t.box = fresh ? { ...b } : {
      x0: t.box.x0 + a * (b.x0 - t.box.x0), y0: t.box.y0 + a * (b.y0 - t.box.y0),
      x1: t.box.x1 + a * (b.x1 - t.box.x1), y1: t.box.y1 + a * (b.y1 - t.box.y1),
    };
    t.boxes = d.boxes; t.facePx = d.facePx; t.rect = d.rect; t.det = d; t.lastSeen = now;
  }

  /** The visible face that most needs a (new) classification, or null. */
  next(now, isEligible = () => true) {
    let best = null;
    for (const t of this.tracks) {
      if (!t.visible || !isEligible(t) || now - t.lastClassified < this.o.minIntervalMs) continue;
      if (!best || t.lastClassified < best.lastClassified) best = t;
    }
    return best;
  }

  /** Record a classifier result (probs: {class: p}) for a track. */
  addSample(t, probs, now) {
    t.samples.push({ probs, at: now });
    t.lastClassified = now;
    const o = this.o;
    t.samples = t.samples.filter((s) => now - s.at <= o.maxAgeMs).slice(-o.window);
    t.probs = meanProbs(t.samples.map((s) => s.probs));
    t.stable = t.samples.length >= o.stableAfter;
  }
}

export function meanProbs(list) {
  if (!list.length) return null;
  const out = {};
  for (const p of list) for (const [k, v] of Object.entries(p)) out[k] = (out[k] || 0) + v / list.length;
  return out;
}

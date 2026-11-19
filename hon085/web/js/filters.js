// See your Life X-ray - signal filters
// One Euro filter (Casiez, Roussel & Vogel, CHI 2012): smooths jitter when you hold still,
// and follows quickly when you move. Used for every tracked point.

class LowPass {
  constructor() { this.y = null; }
  filter(x, a) { this.y = this.y === null ? x : a * x + (1 - a) * this.y; return this.y; }
}

const alpha = (cutoff, dt) => { const tau = 1 / (2 * Math.PI * cutoff); return 1 / (1 + tau / dt); };

export class OneEuro {
  constructor(minCutoff = 1.2, beta = 0.01, dCutoff = 1.0) {
    this.minCutoff = minCutoff; this.beta = beta; this.dCutoff = dCutoff;
    this.x = new LowPass(); this.dx = new LowPass(); this.last = null;
  }
  reset() { this.x = new LowPass(); this.dx = new LowPass(); this.last = null; }
  filter(value, tSec) {
    if (this.last === null) { this.last = tSec; this.dx.filter(0, 1); return this.x.filter(value, 1); }
    const dt = Math.max(1e-3, Math.min(0.5, tSec - this.last)); this.last = tSec;
    const prev = this.x.y;
    const d = this.dx.filter((value - prev) / dt, alpha(this.dCutoff, dt));
    const cutoff = this.minCutoff + this.beta * Math.abs(d);
    return this.x.filter(value, alpha(cutoff, dt));
  }
}

// A bank of 3D One Euro filters for a list of points, with outlier rejection:
// a point that suddenly jumps much further than it can move in one frame is held
// for up to `holdFrames` frames; if the jump persists it is accepted (it was real).
export class PointFilter {
  constructor(n, opts = {}) {
    this.n = n; this.opts = Object.assign({ minCutoff: 1.2, beta: 0.008, maxJump: Infinity, holdFrames: 2 }, opts);
    this.f = Array.from({ length: n * 3 }, () => new OneEuro(this.opts.minCutoff, this.opts.beta));
    this.out = new Float32Array(n * 3); this.prev = null; this.held = new Uint8Array(n);
  }
  setParams(minCutoff, beta) { for (const f of this.f) { f.minCutoff = minCutoff; f.beta = beta; } this.opts.minCutoff = minCutoff; this.opts.beta = beta; }
  reset() { for (const f of this.f) f.reset(); this.prev = null; this.held.fill(0); }
  /** pts: Float32Array(n*3) raw; maxJump: allowed jump (same units) this frame */
  update(pts, tSec, maxJump = this.opts.maxJump) {
    const n = this.n;
    if (!this.prev) this.prev = Float32Array.from(pts);
    for (let i = 0; i < n; i++) {
      const j = i * 3;
      const dx = pts[j] - this.prev[j], dy = pts[j + 1] - this.prev[j + 1];
      const jump = Math.hypot(dx, dy);
      let src = pts;
      if (jump > maxJump && this.held[i] < this.opts.holdFrames) { this.held[i]++; src = this.prev; }
      else this.held[i] = 0;
      for (let k = 0; k < 3; k++) {
        this.prev[j + k] = src[j + k];
        this.out[j + k] = this.f[j + k].filter(src[j + k], tSec);
      }
    }
    return this.out;
  }
}

// Slow scalar smoother (exponential, time based)
export class Smooth {
  constructor(seconds = 0.5) { this.tc = seconds; this.v = null; this.t = null; }
  update(x, tSec) {
    if (this.v === null || !isFinite(this.v)) { this.v = x; this.t = tSec; return x; }
    const dt = Math.max(0, tSec - this.t); this.t = tSec;
    const a = 1 - Math.exp(-dt / this.tc); this.v += (x - this.v) * a; return this.v;
  }
}

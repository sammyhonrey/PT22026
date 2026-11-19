// See your Life X-ray - body rig
// Turns tracked points into 45 segment transforms (pelvis, chest, head, arms, hands, fingers, legs, feet).
// Every anatomy vertex was bound (in tools/build_anatomy.py) to one or two of these segments.
//
// Each segment has a "frame": an origin plus three axes built the same way from the same joints,
// once for the anatomy model at rest and once per video frame for you. The transform for a segment is
//      M = frame(you) * inverse(frame(model at rest))
// so the bone ends land exactly on your tracked joints, and the bone keeps its proportions sideways.
import * as THREE from "three";
import { PointFilter, Smooth } from "./filters.js";

export const NSEG = 45;
export const SEG = { pelvis: 0, chest: 1, head: 2 };
for (const [s, b] of [["L", 3], ["R", 6]]) { SEG[s + "uarm"] = b; SEG[s + "farm"] = b + 1; SEG[s + "palm"] = b + 2; }
for (const [s, b] of [["L", 9], ["R", 12]]) { SEG[s + "thigh"] = b; SEG[s + "shin"] = b + 1; SEG[s + "foot"] = b + 2; }
const FINGERS = { thumb: [1, 2, 3, 4], index: [5, 6, 7, 8], middle: [9, 10, 11, 12], ring: [13, 14, 15, 16], little: [17, 18, 19, 20] };
const FNAMES = Object.keys(FINGERS);
for (const [s, b] of [["L", 15], ["R", 30]]) FNAMES.forEach((f, fi) => { for (let k = 0; k < 3; k++) SEG[s + f + k] = b + fi * 3 + k; });

// MediaPipe pose indices for each side
const SIDE = {
  L: { sh: 11, el: 13, wr: 15, pk: 17, ix: 19, th: 21, hip: 23, kn: 25, an: 27, he: 29, fi: 31 },
  R: { sh: 12, el: 14, wr: 16, pk: 18, ix: 20, th: 22, hip: 24, kn: 26, an: 28, he: 30, fi: 32 },
};
const BONES = [[11, 13], [13, 15], [12, 14], [14, 16], [23, 25], [25, 27], [24, 26], [26, 28]];
const SCALE_PAIRS = [[11, 12], [23, 24], [11, 23], [12, 24], [11, 13], [13, 15], [12, 14], [14, 16], [23, 25], [25, 27], [24, 26], [26, 28]];

const V = () => new THREE.Vector3();
const e1 = V(), e2 = V(), e3 = V(), t1 = V(), t2 = V(), t3 = V(), t4 = V(), UPV = new THREE.Vector3(0, 1, 0);
const sstep = (a, b, x) => { const t = Math.min(1, Math.max(0, (x - a) / (b - a))); return t * t * (3 - 2 * t); };

/** out = basis (a, b orthogonalised, a x b) scaled by (sa, sb, sc), at origin o */
function frame(out, o, a, b, sa, sb, sc) {
  e1.copy(a); if (e1.lengthSq() < 1e-12) e1.set(0, -1, 0); e1.normalize();
  e2.copy(b).addScaledVector(e1, -b.dot(e1));
  if (e2.lengthSq() < 1e-10) { e2.set(1, 0, 0).addScaledVector(e1, -e1.x); if (e2.lengthSq() < 1e-6) e2.set(0, 0, 1).addScaledVector(e1, -e1.z); }
  e2.normalize(); e3.crossVectors(e1, e2);
  out.makeBasis(e1.multiplyScalar(sa), e2.multiplyScalar(sb), e3.multiplyScalar(sc)).setPosition(o);
  return out;
}
const orth = (v, axis) => v.addScaledVector(axis, -v.dot(axis));

/** Builds all frames from points. P: 33 Vector3; H: {L, R} arrays of 21 Vector3 or null. */
function buildFrames(F, P, H, G, Ghead, fit, variant) {
  const midHip = t1.copy(P[23]).add(P[24]).multiplyScalar(0.5);
  const midSh = t2.copy(P[11]).add(P[12]).multiplyScalar(0.5);
  const up = V().subVectors(midSh, midHip);
  const shoulderAcross = V().subVectors(P[11], P[12]), hipAcross = V().subVectors(P[23], P[24]);
  const fwdChest = V().crossVectors(shoulderAcross, up).normalize();
  const fwdPelvis = V().crossVectors(hipAcross, up).normalize();
  const w = fit.size * fit.width, d = fit.size * fit.depth;
  const shift = V().copy(up).normalize().multiplyScalar(fit.shiftY * G);

  frame(F[SEG.pelvis], V().copy(midHip).add(shift), up, hipAcross, up.length(), hipAcross.length() * w, G * d);
  frame(F[SEG.chest], V().copy(midSh).add(shift), up, shoulderAcross, up.length(), shoulderAcross.length() * w, G * d);
  // head: anchored between the eyes (the most reliable face points); eyes over mouth give "up",
  // ears and outer eye corners together give "across"
  const midEye = V().addVectors(P[2], P[5]).multiplyScalar(0.5);
  const headUp = V().copy(midEye).sub(t3.addVectors(P[9], P[10]).multiplyScalar(0.5));
  // "across" from two clues: ear-to-ear / eye-to-eye, and (up x forward) where forward = nose minus mid-ear.
  // The second clue keeps the skull turned the right way when the head is seen from the side.
  const earAcross = V().subVectors(P[7], P[8]).add(t3.subVectors(P[3], P[6])).normalize();
  const fwdHead = V().subVectors(P[0], t3.addVectors(P[7], P[8]).multiplyScalar(0.5));
  const side2 = V().crossVectors(headUp, fwdHead);
  if (side2.lengthSq() > 1e-8) earAcross.add(side2.normalize());
  const hs = Ghead * fit.size * fit.head;
  frame(F[SEG.head], midEye.add(shift), headUp, earAcross, hs, hs, hs);

  for (const s of ["L", "R"]) {
    const I = SIDE[s], h = H && H[s];
    const sh = P[I.sh], el = P[I.el], wr = h ? h[0] : P[I.wr];
    // upper arm: roll from the elbow's bend plane, or from "forward" when the arm is straight
    const ua = V().subVectors(el, sh), uaN = V().copy(ua).normalize();
    const fa = V().subVectors(wr, el), faO = orth(V().copy(fa), uaN);
    const bendA = sstep(0.18, 0.5, faO.length() / Math.max(1e-6, fa.length()));
    const bU = orth(V().copy(fwdChest), uaN).normalize().multiplyScalar(1 - bendA).add(faO.normalize().multiplyScalar(bendA));
    frame(F[SEG[s + "uarm"]], sh, ua, bU, ua.length(), G * w, G * d);
    // forearm: roll from the hand (index knuckle minus little-finger knuckle)
    const across = h ? V().subVectors(h[5], h[17]) : V().subVectors(P[I.ix], P[I.pk]);
    frame(F[SEG[s + "farm"]], el, fa, across, fa.length(), G * w, G * d);
    // palm and fingers
    if (h) {
      const pa = V().subVectors(h[9], h[0]), hw = across.length();
      frame(F[SEG[s + "palm"]], h[0], pa, across, pa.length(), hw, hw);
      for (const f of FNAMES) { const c = FINGERS[f];
        for (let k = 0; k < 3; k++) { const a = V().subVectors(h[c[k + 1]], h[c[k]]);
          frame(F[SEG[s + f + k]], h[c[k]], a, across, a.length(), hw, hw); } }
    } else {
      const knuck = V().addVectors(P[I.ix], P[I.pk]).multiplyScalar(0.5), pa = V().subVectors(knuck, wr);
      frame(F[SEG[s + "palm"]], wr, pa, across, pa.length(), G * w, G * d);
      if (variant === "rest") for (const f of FNAMES) for (let k = 0; k < 3; k++) F[SEG[s + f + k]].copy(F[SEG[s + "palm"]]);
    }
    // legs
    const hip = P[I.hip], kn = P[I.kn], an = P[I.an], he = P[I.he], fi = P[I.fi];
    const th = V().subVectors(kn, hip), thN = V().copy(th).normalize();
    const sh2 = V().subVectors(an, kn), shO = orth(V().copy(sh2), thN);
    const bendK = sstep(0.18, 0.5, shO.length() / Math.max(1e-6, sh2.length()));
    const bT = orth(V().copy(fwdPelvis), thN).normalize().multiplyScalar(1 - bendK).add(shO.normalize().multiplyScalar(-bendK));
    frame(F[SEG[s + "thigh"]], hip, th, bT, th.length(), G * w, G * d);
    const footDir = V().subVectors(fi, he);
    frame(F[SEG[s + "shin"]], kn, sh2, footDir, sh2.length(), G * w, G * d);
    frame(F[SEG[s + "foot"]], he, footDir, V().subVectors(kn, an), footDir.length(), G * w, G * w);
  }
}

export class Rig {
  constructor(manifest) {
    this.restP = manifest.pose_rest.map(p => new THREE.Vector3().fromArray(p));
    this.restH = { L: manifest.hand_rest.L.map(p => new THREE.Vector3().fromArray(p)), R: manifest.hand_rest.R.map(p => new THREE.Vector3().fromArray(p)) };
    this.restG = this.restP[11].distanceTo(this.restP[12]);
    this.restGhead = 1;                                   // head scale is a plain ratio (you / model)
    const rp = this.restP;
    this.restFace = [rp[7].distanceTo(rp[8]), rp[3].distanceTo(rp[6]), rp[2].clone().add(rp[5]).sub(rp[9]).sub(rp[10]).multiplyScalar(0.5).length()];
    this.restTorso = this.restP[11].clone().add(this.restP[12]).sub(this.restP[23]).sub(this.restP[24]).multiplyScalar(0.5).length();
    const unit = { size: 1, width: 1, depth: 1, head: 1, shiftY: 0 };
    const mk = () => Array.from({ length: NSEG }, () => new THREE.Matrix4());
    // rest frames: pose-only hands, and tracked-hand variant
    this.restInvPose = mk(); buildFrames(this.restInvPose, this.restP, null, this.restG, this.restGhead, unit, "rest");
    this.restInvHand = mk(); buildFrames(this.restInvHand, this.restP, this.restH, this.restG, this.restGhead, unit, "rest");
    for (const m of this.restInvPose) m.invert(); for (const m of this.restInvHand) m.invert();
    this.F = mk(); this.M = mk();
    this.segData = new Float32Array(NSEG * 16);           // shared with the shaders
    this.alpha = new Float32Array(NSEG); this.alphaTarget = new Float32Array(NSEG);
    this.lostAt = new Float64Array(NSEG).fill(-1);
    this.P = Array.from({ length: 33 }, V); this.H = { L: Array.from({ length: 21 }, V), R: Array.from({ length: 21 }, V) };
    this.vis = new Float32Array(33); this.raw = new Float32Array(33 * 3); this.worldRaw = new Float32Array(33 * 3);
    this.filter = new PointFilter(33, { minCutoff: 1.0, beta: 0.006, holdFrames: 2 });
    this.worldFilter = new PointFilter(33, { minCutoff: 1.0, beta: 0.4 });
    this.handFilter = { L: new PointFilter(21, { minCutoff: 1.4, beta: 0.01 }), R: new PointFilter(21, { minCutoff: 1.4, beta: 0.01 }) };
    this.handSeen = { L: -1, R: -1 }; this.handRaw = { L: new Float32Array(63), R: new Float32Array(63) };
    this.G = new Smooth(0.35); this.Ghead = new Smooth(0.35); this.pxPerM = new Smooth(0.25);
    this.lengths = BONES.map(() => []); this.lockedLen = BONES.map(() => 0); this.calibrated = false;
    this.world = Array.from({ length: 33 }, V);            // smoothed world points (metres, y up, z toward camera)
    this.found = false; this.lastSeen = -1; this.quality = 0; this.handsTracked = { L: false, R: false };
    this.opts = { boneLock: true, smooth: 0.5 };
    this.fit = { size: 1, width: 1, depth: 1, head: 1, shiftY: 0 };
    this.setSmoothing(0.5);
  }

  setSmoothing(v) { // 0 = fastest, 1 = smoothest
    this.opts.smooth = v;
    const mc = 3.2 * Math.pow(0.12, v);          // 3.2 Hz .. 0.38 Hz
    this.filter.setParams(mc, 0.004 + 0.01 * (1 - v));
    for (const s of ["L", "R"]) this.handFilter[s].setParams(mc * 1.3, 0.006 + 0.012 * (1 - v));
  }

  resetCalibration() { this.lengths = BONES.map(() => []); this.lockedLen = BONES.map(() => 0); this.calibrated = false; }

  /** Collects limb lengths for `seconds` while the person stands in view. */
  calibrateFromWorld() {
    BONES.forEach(([a, b], i) => { const L = this.lengths[i]; if (L.length >= 15) { const s = [...L].sort((x, y) => x - y); this.lockedLen[i] = s[s.length >> 1]; } });
    this.calibrated = this.lockedLen.every(x => x > 0);
    return this.calibrated;
  }

  /** One new tracker result. W, H: picture size in pixels. t: seconds. */
  update(res, W, H, t) {
    if (!res || !res.pose) { this.markLost(t); return; }
    const lm = res.pose, wl = res.world;
    for (let i = 0; i < 33; i++) {
      const p = lm[i], v = p.visibility === undefined ? 1 : p.visibility;
      const m = (i === 11 || i === 12 || i === 23 || i === 24) ? 0.2 : 0.03;      // shoulders and hips may sit just outside the picture
      const inFrame = p.x > -m && p.x < 1 + m && p.y > -m && p.y < 1 + m;
      this.vis[i] = this.vis[i] * 0.5 + (inFrame ? v : v * 0.3) * 0.5;
      this.raw[i * 3] = p.x * W; this.raw[i * 3 + 1] = (1 - p.y) * H;
      if (wl) { this.worldRaw[i * 3] = wl[i].x; this.worldRaw[i * 3 + 1] = -wl[i].y; this.worldRaw[i * 3 + 2] = -wl[i].z; }
    }
    const wf = this.worldFilter.update(this.worldRaw, t);
    for (let i = 0; i < 33; i++) this.world[i].fromArray(wf, i * 3);
    // pixels per metre: compare picture lengths with real-world lengths (median over the visible limbs)
    const ratios = [];
    for (const [a, b] of SCALE_PAIRS) {
      if (this.vis[a] < 0.6 || this.vis[b] < 0.6) continue;
      const dw = Math.hypot(this.worldRaw[a * 3] - this.worldRaw[b * 3], this.worldRaw[a * 3 + 1] - this.worldRaw[b * 3 + 1]);
      if (dw < 0.08) continue;
      ratios.push(Math.hypot(this.raw[a * 3] - this.raw[b * 3], this.raw[a * 3 + 1] - this.raw[b * 3 + 1]) / dw);
    }
    if (ratios.length) { ratios.sort((x, y) => x - y); this.pxPerM.update(ratios[ratios.length >> 1], t); }
    const s = this.pxPerM.v || W * 0.6;
    for (let i = 0; i < 33; i++) this.raw[i * 3 + 2] = this.worldRaw[i * 3 + 2] * s;

    // limb lengths: learn them automatically (or after "Calibrate"), then keep each bone the same length
    BONES.forEach(([a, b], i) => {
      if (this.vis[a] > 0.85 && this.vis[b] > 0.85) {
        const L = this.lengths[i]; const d = Math.hypot(this.worldRaw[a * 3] - this.worldRaw[b * 3], this.worldRaw[a * 3 + 1] - this.worldRaw[b * 3 + 1], this.worldRaw[a * 3 + 2] - this.worldRaw[b * 3 + 2]);
        if (L.length < 240) L.push(d); else L[(L.length * Math.random()) | 0] = d;
      }
    });
    if (!this.calibrated && this.lengths.every(L => L.length >= 45)) this.calibrateFromWorld();
    if (this.opts.boneLock && this.calibrated) {
      for (let i = 0; i < BONES.length; i++) {
        const [a, b] = BONES[i], Lpx = this.lockedLen[i] * s;
        const d2 = Math.hypot(this.raw[a * 3] - this.raw[b * 3], this.raw[a * 3 + 1] - this.raw[b * 3 + 1]);
        const dzW = this.raw[b * 3 + 2] - this.raw[a * 3 + 2];
        let dz = d2 < Lpx ? Math.sqrt(Lpx * Lpx - d2 * d2) : 0;
        dz = Math.min(dz, Math.abs(dzW) * 1.6 + Lpx * 0.15) * Math.sign(dzW || 1);
        this.raw[b * 3 + 2] = this.raw[a * 3 + 2] + dz;
      }
    }
    // sitting at a desk: hips are out of the picture, so place them under the shoulders
    const hipsOk = this.vis[23] > 0.5 && this.vis[24] > 0.5;
    if (!hipsOk) {
      const ax = this.raw[33] - this.raw[36], ay = this.raw[34] - this.raw[37];         // left minus right shoulder
      const len = Math.hypot(ax, ay) || 1, dnx = ay / len, dny = -ax / len;            // perpendicular, pointing down
      const torso = (this.restTorso / this.restG) * len;
      const mx = (this.raw[33] + this.raw[36]) / 2, my = (this.raw[34] + this.raw[37]) / 2, mz = (this.raw[35] + this.raw[38]) / 2;
      const hw = len * 0.3;
      const k = 1 - Math.max(this.vis[23], this.vis[24]);                               // blend toward the estimate
      for (const [i, sg] of [[23, 1], [24, -1]]) {
        const ex = mx + dnx * torso + sg * ax / len * hw, ey = my + dny * torso + sg * ay / len * hw;
        this.raw[i * 3] += (ex - this.raw[i * 3]) * k; this.raw[i * 3 + 1] += (ey - this.raw[i * 3 + 1]) * k; this.raw[i * 3 + 2] += (mz - this.raw[i * 3 + 2]) * k;
      }
    }
    // hands: match each tracked hand to the nearer wrist of the body
    const shoulderPx = Math.hypot(this.raw[33] - this.raw[36], this.raw[34] - this.raw[37]) || W * 0.2;
    const used = { L: false, R: false };
    for (const hd of res.hands || []) {
      const hx = hd.lm[0].x * W, hy = (1 - hd.lm[0].y) * H;
      const dL = Math.hypot(hx - this.raw[45], hy - this.raw[46]), dR = Math.hypot(hx - this.raw[48], hy - this.raw[49]);
      const side = dL < dR ? "L" : "R", d = Math.min(dL, dR);
      if (used[side] || d > shoulderPx * 0.9) continue;
      used[side] = true; const wi = side === "L" ? 15 : 16, buf = this.handRaw[side];
      for (let j = 0; j < 21; j++) {
        buf[j * 3] = hd.lm[j].x * W; buf[j * 3 + 1] = (1 - hd.lm[j].y) * H; buf[j * 3 + 2] = this.raw[wi * 3 + 2] - hd.lm[j].z * W;
      }
      this.raw[wi * 3] = buf[0]; this.raw[wi * 3 + 1] = buf[1];                         // the hand tracker's wrist is more exact
      const hf = this.handFilter[side];
      if (this.handSeen[side] < 0 || t - this.handSeen[side] > 0.3) hf.reset();
      const o = hf.update(buf, t); for (let j = 0; j < 21; j++) this.H[side][j].fromArray(o, j * 3);
      this.handSeen[side] = t;
    }
    for (const sd of ["L", "R"]) this.handsTracked[sd] = this.handSeen[sd] >= 0 && t - this.handSeen[sd] < 0.25;

    if (!this.found || t - this.lastSeen > 0.6) this.filter.reset();
    const o = this.filter.update(this.raw, t, shoulderPx * 0.45);
    for (let i = 0; i < 33; i++) this.P[i].fromArray(o, i * 3);
    this.G.update(this.P[11].distanceTo(this.P[12]), t);
    const P = this.P, d2 = (a, b) => Math.hypot(P[a].x - P[b].x, P[a].y - P[b].y);
    const eyeMouth = Math.hypot((P[2].x + P[5].x - P[9].x - P[10].x) / 2, (P[2].y + P[5].y - P[9].y - P[10].y) / 2);
    const hr = Math.max(d2(7, 8) / this.restFace[0], d2(3, 6) / this.restFace[1], eyeMouth / this.restFace[2]);
    this.Ghead.update(this.vis[0] > 0.5 ? hr : this.G.v / this.restG, t);
    this.found = true; this.lastSeen = t;
    this.quality = (this.vis[0] + this.vis[11] + this.vis[12]) / 3;      // head and shoulders are the anchor of the whole fit
    this.solve(t);
  }

  markLost(t) {
    if (this.found && t - this.lastSeen > 0.5) this.found = false;
    for (let i = 0; i < NSEG; i++) if (this.lostAt[i] < 0) this.lostAt[i] = t;
    this.quality *= 0.9;
  }

  solve(t) {
    const H = { L: this.handsTracked.L ? this.H.L : null, R: this.handsTracked.R ? this.H.R : null };
    buildFrames(this.F, this.P, H, this.G.v, this.Ghead.v, this.fit, "live");
    const v = this.vis, ok = (...ids) => ids.every(i => v[i] > 0.5);
    const good = new Array(NSEG).fill(false);
    good[SEG.chest] = ok(11, 12); good[SEG.pelvis] = ok(11, 12);
    good[SEG.head] = v[0] > 0.5 && (v[7] > 0.3 || v[8] > 0.3);
    for (const s of ["L", "R"]) {
      const I = SIDE[s];
      good[SEG[s + "uarm"]] = ok(I.sh, I.el); good[SEG[s + "farm"]] = ok(I.el, I.wr); good[SEG[s + "palm"]] = ok(I.wr) || !!H[s];
      good[SEG[s + "thigh"]] = ok(I.hip, I.kn); good[SEG[s + "shin"]] = ok(I.kn, I.an); good[SEG[s + "foot"]] = ok(I.an, I.he, I.fi);
      for (const f of FNAMES) for (let k = 0; k < 3; k++) good[SEG[s + f + k]] = good[SEG[s + "palm"]];
    }
    for (const s of ["L", "R"]) {
      const inv = H[s] ? this.restInvHand : this.restInvPose;
      for (const name of ["farm", "palm"]) { const i = SEG[s + name]; this.M[i].multiplyMatrices(this.F[i], inv[i]); }
      for (const f of FNAMES) for (let k = 0; k < 3; k++) {
        const i = SEG[s + f + k];
        if (H[s]) this.M[i].multiplyMatrices(this.F[i], this.restInvHand[i]);
        else this.M[i].copy(this.M[SEG[s + "palm"]]);
      }
    }
    for (let i = 0; i < NSEG; i++) {
      const isHandPart = i >= 15 || i === 4 || i === 5 || i === 7 || i === 8;
      if (good[i]) {
        if (!isHandPart) this.M[i].multiplyMatrices(this.F[i], this.restInvPose[i]);
        this.M[i].toArray(this.segData, i * 16);
        this.lostAt[i] = -1; this.alphaTarget[i] = 1;
      } else {
        // keep the last good position for a moment, then fade out (no popping on brief tracking gaps)
        if (this.lostAt[i] < 0) this.lostAt[i] = t;
        this.alphaTarget[i] = t - this.lostAt[i] < 0.35 ? this.alphaTarget[i] : 0;
        if (this.alpha[i] === 0) { this.M[i].multiplyMatrices(this.F[i], (isHandPart && H[i < 30 && i >= 15 ? "L" : "R"]) ? this.restInvHand[i] : this.restInvPose[i]); this.M[i].toArray(this.segData, i * 16); }
      }
    }
  }

  /** Called every displayed frame: eases segment opacity toward its target. */
  tick(dt, t) {
    const gone = !this.found || t - this.lastSeen > 0.35;
    for (let i = 0; i < NSEG; i++) {
      const target = gone ? 0 : this.alphaTarget[i];
      const rate = target > this.alpha[i] ? 6 : 3.5;
      this.alpha[i] += (target - this.alpha[i]) * Math.min(1, dt * rate);
      if (this.alpha[i] < 0.004) this.alpha[i] = 0;
    }
  }

  /** Position (pixels) of a rest-space point bound to segments s0/s1 with weight w. */
  place(out, p, s0, s1, w) {
    t4.set(p[0], p[1], p[2]);
    const a = t3.copy(t4).applyMatrix4(this.M[s0]), b = t4.applyMatrix4(this.M[s1]);
    return out.copy(a).multiplyScalar(w).addScaledVector(b, 1 - w);
  }
}

// ----------------------------------------------------------------------------- measurements
const ang = (a, b, c) => { const u = V().subVectors(a, b), w = V().subVectors(c, b); return THREE.MathUtils.radToDeg(u.angleTo(w)); };
/** Joint angles and posture from smoothed real-world points (degrees). */
export function measure(rig) {
  const W = rig.world, v = rig.vis, ok = (...i) => i.every(k => v[k] > 0.6);
  const out = {};
  const midHip = V().addVectors(W[23], W[24]).multiplyScalar(0.5), midSh = V().addVectors(W[11], W[12]).multiplyScalar(0.5);
  const trunkDown = V().subVectors(midHip, midSh);
  out.l_elbow = ok(11, 13, 15) ? 180 - ang(W[11], W[13], W[15]) : null;
  out.r_elbow = ok(12, 14, 16) ? 180 - ang(W[12], W[14], W[16]) : null;
  out.l_shoulder = ok(11, 13) ? THREE.MathUtils.radToDeg(V().subVectors(W[13], W[11]).angleTo(trunkDown)) : null;
  out.r_shoulder = ok(12, 14) ? THREE.MathUtils.radToDeg(V().subVectors(W[14], W[12]).angleTo(trunkDown)) : null;
  out.l_knee = ok(23, 25, 27) ? 180 - ang(W[23], W[25], W[27]) : null;
  out.r_knee = ok(24, 26, 28) ? 180 - ang(W[24], W[26], W[28]) : null;
  out.l_hip = ok(11, 23, 25) ? 180 - ang(W[11], W[23], W[25]) : null;
  out.r_hip = ok(12, 24, 26) ? 180 - ang(W[12], W[24], W[26]) : null;
  out.trunk_lean = ok(11, 12, 23, 24) ? THREE.MathUtils.radToDeg(V().subVectors(midSh, midHip).angleTo(UPV)) : null;
  out.shoulder_tilt = ok(11, 12) ? THREE.MathUtils.radToDeg(Math.atan2(W[11].y - W[12].y, Math.hypot(W[11].x - W[12].x, W[11].z - W[12].z))) : null;
  out.head_tilt = ok(7, 8) ? THREE.MathUtils.radToDeg(Math.atan2(W[7].y - W[8].y, Math.hypot(W[7].x - W[8].x, W[7].z - W[8].z))) : null;
  out.head_turn = ok(7, 8) ? THREE.MathUtils.radToDeg(Math.atan2(W[7].z - W[8].z, W[7].x - W[8].x)) : null;
  out.shoulder_width_cm = ok(11, 12) ? W[11].distanceTo(W[12]) * 100 : null;
  return out;
}

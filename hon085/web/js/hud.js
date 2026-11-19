// See your Life X-ray - heads-up layer: anatomy labels, tracking points, x-ray lens ring
import * as THREE from "three";
import { LAYERS } from "./scene.js";

const POSE_LINKS = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28],
                    [27, 29], [29, 31], [27, 31], [28, 30], [30, 32], [28, 32], [15, 17], [15, 19], [17, 19], [16, 18], [16, 20], [18, 20], [0, 7], [0, 8], [9, 10]];
const HAND_LINKS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]];
const COLOR = Object.fromEntries(LAYERS.map(l => [l.id, l.color]));
const tmp = new THREE.Vector3();

export class Hud {
  constructor(canvas) { this.c = canvas; this.g = canvas.getContext("2d"); this.state = new Map(); this.frame = 0; }
  resize(W, H, dpr) { this.W = W; this.H = H; this.dpr = dpr; this.c.width = Math.round(W * dpr); this.c.height = Math.round(H * dpr); }
  sx(x, mirror) { return mirror ? this.W - x : x; }

  draw(rig, labels, S, mirror, lens) {
    const g = this.g; this.frame++;
    g.setTransform(this.dpr, 0, 0, this.dpr, 0, 0); g.clearRect(0, 0, this.W, this.H);
    if (lens) {
      g.save(); g.strokeStyle = "rgba(94,234,212,.85)"; g.lineWidth = 1.5; g.setLineDash([6, 6]);
      g.beginPath(); g.arc(lens.x, lens.y, lens.r, 0, Math.PI * 2); g.stroke(); g.restore();
    }
    if (S.points && rig.found) this.drawPoints(rig, mirror);
    if (S.labels && rig.found) this.drawLabels(rig, labels, S, mirror, lens);
  }

  drawPoints(rig, mirror) {
    const g = this.g, P = rig.P, v = rig.vis, H = this.H;
    g.lineWidth = 2; g.strokeStyle = "rgba(94,234,212,.75)";
    for (const [a, b] of POSE_LINKS) {
      if (v[a] < 0.5 || v[b] < 0.5) continue;
      g.beginPath(); g.moveTo(this.sx(P[a].x, mirror), H - P[a].y); g.lineTo(this.sx(P[b].x, mirror), H - P[b].y); g.stroke();
    }
    for (let i = 0; i < 33; i++) {
      if (v[i] < 0.5) continue;
      g.fillStyle = i < 11 ? "#c4b5fd" : i % 2 ? "#5eead4" : "#fbbf24";
      g.beginPath(); g.arc(this.sx(P[i].x, mirror), H - P[i].y, 3.2, 0, 6.283); g.fill();
    }
    for (const s of ["L", "R"]) {
      if (!rig.handsTracked[s]) continue;
      const h = rig.H[s]; g.strokeStyle = "rgba(251,191,36,.8)"; g.lineWidth = 1.5;
      for (const [a, b] of HAND_LINKS) { g.beginPath(); g.moveTo(this.sx(h[a].x, mirror), H - h[a].y); g.lineTo(this.sx(h[b].x, mirror), H - h[b].y); g.stroke(); }
      g.fillStyle = "#fde68a"; for (const p of h) { g.beginPath(); g.arc(this.sx(p.x, mirror), H - p.y, 2.2, 0, 6.283); g.fill(); }
    }
  }

  drawLabels(rig, labels, S, mirror, lens) {
    const g = this.g, H = this.H, W = this.W;
    // one anchor per label text: the one on a well-tracked segment, nearest the camera
    const best = new Map();
    for (const L of labels) {
      if (!S.layers[L.layer]) continue;
      const a = rig.alpha[L.s[0]] * L.w + rig.alpha[L.s[1]] * (1 - L.w);
      if (a < 0.6) continue;
      rig.place(tmp, L.p, L.s[0], L.s[1], L.w);
      const x = this.sx(tmp.x, mirror), y = H - tmp.y;
      if (x < 4 || x > W - 4 || y < 4 || y > H - 4) continue;
      if (lens && Math.hypot(x - lens.x, y - lens.y) > lens.r) continue;
      const prev = best.get(L.t);
      if (!prev || tmp.z > prev.z) best.set(L.t, { text: L.t, layer: L.layer, ax: x, ay: y, z: tmp.z });
    }
    let items = [...best.values()];
    // keep it readable: at most 6 labels per side, organs first
    const pri = { heart: 0, lungs: 1, organs: 2, brain: 3, vessels: 4, skeleton: 5, muscles: 6 };
    items.sort((p, q) => (pri[p.layer] ?? 9) - (pri[q.layer] ?? 9));
    // body centre on screen
    const P = rig.P, cx = this.sx((P[11].x + P[12].x + P[23].x + P[24].x) / 4, mirror);
    const halfW = Math.abs(P[11].x - P[12].x) * 0.95 + 30;
    const cols = { left: [], right: [] };
    for (const it of items) {
      const st = this.state.get(it.text) || { side: it.ax < cx ? "left" : "right", y: it.ay, seen: 0 };
      if (st.side === "left" && it.ax > cx + halfW * 0.25) st.side = "right"; else if (st.side === "right" && it.ax < cx - halfW * 0.25) st.side = "left";
      if (this.frame - st.seen > 4) st.y = it.ay; else st.y += (it.ay - st.y) * 0.35;
      st.seen = this.frame; this.state.set(it.text, st); it.st = st;
      if (cols[st.side].length < S.maxLabels / 2) cols[st.side].push(it);
    }
    g.font = "600 12px Inter, 'Segoe UI', system-ui, sans-serif"; g.textBaseline = "middle";
    for (const side of ["left", "right"]) {
      const list = cols[side].sort((p, q) => p.st.y - q.st.y), gap = 24;
      let prev = -1e9; for (const it of list) { it.ty = Math.max(it.st.y, prev + gap); prev = it.ty; }
      const over = prev - (H - 16); if (over > 0) for (const it of list) it.ty -= over;
      const tx = side === "left" ? Math.max(12, cx - halfW - 40) : Math.min(W - 12, cx + halfW + 40);
      for (const it of list) {
        const col = COLOR[it.layer] || "#fff", tw = g.measureText(it.text).width, pad = 8;
        const bx = side === "left" ? Math.max(4, tx - tw - pad * 2 - 10) : Math.min(W - tw - pad * 2 - 14, tx), by = it.ty;
        const ex = side === "left" ? bx + tw + pad * 2 + 10 : bx;
        g.strokeStyle = col; g.globalAlpha = 0.7; g.lineWidth = 1;
        g.beginPath(); g.moveTo(it.ax, it.ay); g.lineTo((it.ax + ex) / 2, by); g.lineTo(ex, by); g.stroke();
        g.globalAlpha = 1; g.fillStyle = col; g.beginPath(); g.arc(it.ax, it.ay, 3, 0, 6.283); g.fill();
        g.fillStyle = "rgba(6,10,18,.78)"; roundRect(g, bx, by - 11, tw + pad * 2 + 10, 22, 11); g.fill();
        g.strokeStyle = col; g.globalAlpha = 0.55; g.stroke(); g.globalAlpha = 1;
        g.fillStyle = col; g.beginPath(); g.arc(bx + pad + 2, by, 3, 0, 6.283); g.fill();
        g.fillStyle = "#eef4ff"; g.textAlign = "left"; g.fillText(it.text, bx + pad + 10, by + 0.5);
      }
    }
  }
}

function roundRect(g, x, y, w, h, r) {
  g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}

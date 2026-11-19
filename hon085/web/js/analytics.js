// See your Life X-ray - live measurements, chart and data export
export const METRICS = [
  { key: "l_elbow", name: "Left elbow bend", unit: "°" },
  { key: "r_elbow", name: "Right elbow bend", unit: "°" },
  { key: "l_shoulder", name: "Left arm raise", unit: "°" },
  { key: "r_shoulder", name: "Right arm raise", unit: "°" },
  { key: "l_hip", name: "Left hip bend", unit: "°" },
  { key: "r_hip", name: "Right hip bend", unit: "°" },
  { key: "l_knee", name: "Left knee bend", unit: "°" },
  { key: "r_knee", name: "Right knee bend", unit: "°" },
  { key: "trunk_lean", name: "Trunk lean from vertical", unit: "°" },
  { key: "shoulder_tilt", name: "Shoulder tilt", unit: "°", dp: 1 },
  { key: "head_tilt", name: "Head tilt", unit: "°", dp: 1 },
  { key: "head_turn", name: "Head turn", unit: "°" },
  { key: "shoulder_width_cm", name: "Shoulder width (estimate)", unit: " cm", dp: 1 },
];
const NAMES = ["nose", "l_eye_inner", "l_eye", "l_eye_outer", "r_eye_inner", "r_eye", "r_eye_outer", "l_ear", "r_ear", "mouth_l", "mouth_r",
  "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_wrist", "r_wrist", "l_pinky", "r_pinky", "l_index", "r_index", "l_thumb", "r_thumb",
  "l_hip", "r_hip", "l_knee", "r_knee", "l_ankle", "r_ankle", "l_heel", "r_heel", "l_foot_index", "r_foot_index"];

export class DataLog {
  constructor() { this.hist = []; this.rows = []; this.recording = false; this.t0 = 0; }
  push(now, m) { this.hist.push([now, m]); while (this.hist.length && now - this.hist[0][0] > 12000) this.hist.shift(); }
  series(key) { return this.hist.map(([t, m]) => [t, m[key]]); }
  start() { this.rows = []; this.recording = true; this.t0 = performance.now(); }
  stop() { this.recording = false; }
  duration() { return this.rows.length ? (this.rows[this.rows.length - 1].t - this.rows[0].t) / 1000 : 0; }
  add(now, rig, m, fps, inferMs) {
    const pts = new Float32Array(33 * 4);
    for (let i = 0; i < 33; i++) { const w = rig.world[i]; pts[i * 4] = w.x; pts[i * 4 + 1] = w.y; pts[i * 4 + 2] = w.z; pts[i * 4 + 3] = rig.vis[i]; }
    this.rows.push({ t: now - this.t0, q: rig.quality, fps, inferMs, m: { ...m }, pts, hands: (rig.handsTracked.L ? "L" : "") + (rig.handsTracked.R ? "R" : "") });
    if (this.rows.length > 108000) this.stop();       // about one hour at 30 frames per second
  }
  csv() {
    const head = ["time_s", "tracking_quality", "display_fps", "tracker_ms", "hands_tracked", ...METRICS.map(m => m.key)];
    for (const n of NAMES) head.push(n + "_x_m", n + "_y_m", n + "_z_m", n + "_visibility");
    const lines = [head.join(",")];
    for (const r of this.rows) {
      const row = [(r.t / 1000).toFixed(3), r.q.toFixed(3), r.fps, r.inferMs.toFixed(1), r.hands || "-", ...METRICS.map(m => r.m[m.key] == null ? "" : r.m[m.key].toFixed(2))];
      for (let i = 0; i < 33 * 4; i++) row.push(r.pts[i].toFixed(4));
      lines.push(row.join(","));
    }
    return lines.join("\n");
  }
  report(engine) {
    const out = { app: "See your Life X-ray", author: "Mohammad Kazim Honarjo", created: new Date().toISOString(), engine,
      frames: this.rows.length, duration_s: +this.duration().toFixed(2),
      coordinate_system: "metres, origin between the hips, x = to the camera's right, y = up, z = toward the camera",
      note: "Angles come from camera-based pose estimation. Typical error is several degrees and grows when a limb points at the camera or is hidden. Not for medical use.",
      tracking_quality: stats(this.rows.map(r => r.q)), tracker_ms: stats(this.rows.map(r => r.inferMs)), measurements: {} };
    for (const m of METRICS) { const s = stats(this.rows.map(r => r.m[m.key]).filter(v => v != null)); if (s) out.measurements[m.key] = { name: m.name, unit: m.unit.trim(), ...s }; }
    return out;
  }
}

function stats(a) {
  if (!a.length) return null;
  const s = [...a].sort((x, y) => x - y), mean = a.reduce((p, q) => p + q, 0) / a.length;
  const sd = Math.sqrt(a.reduce((p, q) => p + (q - mean) ** 2, 0) / a.length);
  const r = v => +v.toFixed(3);
  return { samples: a.length, min: r(s[0]), p05: r(s[Math.floor(s.length * 0.05)]), median: r(s[s.length >> 1]), p95: r(s[Math.floor(s.length * 0.95)]), max: r(s[s.length - 1]), mean: r(mean), sd: r(sd), range: r(s[s.length - 1] - s[0]) };
}

export function drawChart(canvas, series, meta) {
  const g = canvas.getContext("2d"), W = canvas.width, H = canvas.height;
  g.clearRect(0, 0, W, H);
  const pts = series.filter(p => p[1] != null);
  g.strokeStyle = "rgba(120,160,220,.14)"; g.lineWidth = 1;
  for (let i = 1; i < 4; i++) { g.beginPath(); g.moveTo(0, H * i / 4); g.lineTo(W, H * i / 4); g.stroke(); }
  g.fillStyle = "#8796b4"; g.font = "22px system-ui"; g.textBaseline = "top";
  if (pts.length < 2) { g.fillText((meta ? meta.name : "") + ": waiting for data…", 14, 12); return; }
  let lo = Math.min(...pts.map(p => p[1])), hi = Math.max(...pts.map(p => p[1]));
  if (hi - lo < 10) { const c = (hi + lo) / 2; lo = c - 5; hi = c + 5; }
  const t1 = series[series.length - 1][0], t0 = t1 - 12000;
  const X = t => (t - t0) / 12000 * W, Y = v => H - 14 - (v - lo) / (hi - lo) * (H - 40);
  const grad = g.createLinearGradient(0, 0, W, 0); grad.addColorStop(0, "#5eead4"); grad.addColorStop(1, "#f472b6");
  g.strokeStyle = grad; g.lineWidth = 3; g.lineJoin = "round"; g.beginPath();
  let first = true; for (const [t, v] of pts) { if (first) g.moveTo(X(t), Y(v)); else g.lineTo(X(t), Y(v)); first = false; } g.stroke();
  const last = pts[pts.length - 1][1];
  g.fillStyle = "#e8f0ff"; g.fillText(meta.name + "  " + last.toFixed(meta.dp ?? 0) + meta.unit, 14, 10);
  g.fillStyle = "#5b6a88"; g.fillText(hi.toFixed(0), W - 60, 10); g.fillText(lo.toFixed(0), W - 60, H - 32);
}

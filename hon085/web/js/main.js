// See your Life X-ray - application
// Author: Mohammad Kazim Honarjo
import { Tracker } from "./tracker.js";
import { Rig, measure, NSEG } from "./rig.js";
import { Scene, LAYERS } from "./scene.js";
import { Hud } from "./hud.js";
import { Badge } from "./badge.js";
import { DataLog, METRICS, drawChart } from "./analytics.js";
import { Coach } from "./coach.js";

const $ = id => document.getElementById(id);
const params = new URLSearchParams(location.search);
const TEST = params.has("test");

// ----------------------------------------------------------------------------- settings
const DEFAULTS = {
  layers: { skin: true, skeleton: true, muscles: true, heart: true, vessels: false, lungs: false, organs: false, brain: false },
  look: "xray", bright: 1, opacity: 0.95, real: 0.3, dim: 0.3, mask: true, feather: 0.35, reveal: "whole", cut: 0.5,
  labels: true, maxLabels: 10, heartbeat: true, points: false, readout: false, mirror: true,
  model: "full", hands: true, boneLock: true, smooth: 0.5, camera: "", res: "1280",
  fit: { size: 1, width: 1, depth: 1, head: 1, shiftY: 0 }, coach: true, coachMuscles: true,
};
const S = structuredClone(DEFAULTS);
try { const saved = JSON.parse(localStorage.getItem("slx_settings") || "{}"); Object.assign(S, saved, { layers: { ...DEFAULTS.layers, ...(saved.layers || {}) }, fit: { ...DEFAULTS.fit, ...(saved.fit || {}) } }); } catch (e) {}
let saveTimer = 0;
const save = () => { clearTimeout(saveTimer); saveTimer = setTimeout(() => { try { localStorage.setItem("slx_settings", JSON.stringify(S)); } catch (e) {} }, 250); };

// ----------------------------------------------------------------------------- objects
const stage = $("stage"), video = $("video");
const scene = new Scene($("gl")), hud = new Hud($("hud")), badge = new Badge($("badge"));
const tracker = new Tracker();
let rig = null, labels = [], running = false, source = null, stream = null, recorder = null;
const log = new DataLog();
let coach = null;

// ----------------------------------------------------------------------------- UI helpers
function toast(msg, bad = false, ms = 3800) {
  const t = $("toast"); t.textContent = msg; t.className = "toast" + (bad ? " bad" : ""); t.hidden = false;
  clearTimeout(toast.h); toast.h = setTimeout(() => (t.hidden = true), ms);
}
function loader(text, frac) {
  const l = $("loader"); if (text === null) { l.hidden = true; return; }
  l.hidden = false; $("loaderText").textContent = text; if (frac !== undefined) $("loaderBar").style.width = Math.round(frac * 100) + "%";
}
function setChip(state, text) { const c = $("chipTrack"); c.className = "chip " + state; c.querySelector("b").textContent = text; }

// layer list
const counts = {};
function buildLayerList() {
  const box = $("layerList"); box.innerHTML = "";
  for (const L of LAYERS) {
    const el = document.createElement("label"); el.className = "layer" + (S.layers[L.id] ? " on" : "");
    el.innerHTML = `<input type="checkbox"><span class="sw" style="background:${L.color};color:${L.color}"></span><span class="nm">${L.name}</span><span class="ct">${counts[L.id] || ""}</span><span class="tg"></span>`;
    const cb = el.querySelector("input"); cb.checked = !!S.layers[L.id];
    cb.onchange = () => { S.layers[L.id] = cb.checked; el.classList.toggle("on", cb.checked); scene.applyLook(S); save(); };
    box.appendChild(el);
  }
}
const PRESETS = {
  coach: { skin: true, skeleton: true, muscles: true },
  skeleton: { skin: true, skeleton: true },
  muscles: { skin: true, muscles: true },
  heart: { skin: true, heart: true, vessels: true, lungs: false },
  organs: { skin: true, heart: true, lungs: true, organs: true, brain: true },
  all: { skin: true, skeleton: true, muscles: true, heart: true, vessels: true, lungs: true, organs: true, brain: true },
};
for (const b of document.querySelectorAll("[data-preset]")) b.onclick = () => {
  const p = PRESETS[b.dataset.preset]; for (const L of LAYERS) S.layers[L.id] = !!p[L.id];
  if (b.dataset.preset === "all" && S.look === "anatomy") S.opacity = Math.min(S.opacity, 0.9);
  buildLayerList(); syncUI(); scene.applyLook(S); save();
};

// tabs
for (const b of document.querySelectorAll(".tabs button")) b.onclick = () => {
  for (const x of document.querySelectorAll(".tabs button")) x.classList.toggle("on", x === b);
  for (const p of document.querySelectorAll(".pane")) p.classList.toggle("on", p.dataset.pane === b.dataset.tab);
};

// sliders and switches
const pct = v => Math.round(v * 100) + "%";
const SLIDERS = {
  bright: [() => S.bright, v => (S.bright = v), pct], opacity: [() => S.opacity, v => (S.opacity = v), pct],
  real: [() => S.real, v => (S.real = v), pct], dim: [() => S.dim, v => (S.dim = v), pct], feather: [() => S.feather, v => (S.feather = v), pct],
  cut: [() => S.cut, v => (S.cut = v), null], maxLabels: [() => S.maxLabels, v => (S.maxLabels = v), v => String(v)],
  smooth: [() => S.smooth, v => { S.smooth = v; rig && rig.setSmoothing(v); }, pct],
  fsize: [() => S.fit.size, v => (S.fit.size = v), pct], fwidth: [() => S.fit.width, v => (S.fit.width = v), pct],
  fdepth: [() => S.fit.depth, v => (S.fit.depth = v), pct], fhead: [() => S.fit.head, v => (S.fit.head = v), pct],
  fshift: [() => S.fit.shiftY, v => (S.fit.shiftY = v), v => (v > 0 ? "+" : "") + Math.round(v * 100)],
};
for (const [id, [get, set, fmt]] of Object.entries(SLIDERS)) $(id).oninput = e => { set(parseFloat(e.target.value)); if (fmt) $(id + "V").textContent = fmt(get()); afterChange(); };
const SWITCHES = { coachOn: "coach", coachMuscles: "coachMuscles", labels: "labels", heartbeat: "heartbeat", mask: "mask", points: "points", readoutOn: "readout", hands: "hands", boneLock: "boneLock" };
for (const [id, key] of Object.entries(SWITCHES)) $(id).onchange = e => { S[key] = e.target.checked; afterChange(); };
$("coachMuscles").addEventListener("change", e => { if (e.target.checked && !S.layers.muscles) { S.layers.muscles = true; buildLayerList(); afterChange(); toast("Muscle layer switched on"); } });
$("reveal").onchange = e => { S.reveal = e.target.value; S.cut = 0.5; syncUI(); afterChange(); };
$("model").onchange = async e => {
  S.model = e.target.value; save();
  if (!tracker.pose) return;
  loader("Switching body model…", 0.5);
  try { await tracker.setPoseModel(S.model); toast("Body model: " + S.model); } catch (err) { toast("Could not load that model: " + err.message, true); }
  loader(null); updateEngineChip();
};
$("resSel").onchange = e => { S.res = e.target.value; save(); if (source === "camera") startCamera(); };
$("camSel").onchange = e => { S.camera = e.target.value; save(); if (source === "camera") startCamera(); };
for (const b of $("lookSeg").children) b.onclick = () => { S.look = b.dataset.look; syncUI(); afterChange(); };
$("btnFitReset").onclick = () => { S.fit = { ...DEFAULTS.fit }; syncUI(); afterChange(); };

function afterChange() {
  if (rig) { rig.opts.boneLock = S.boneLock; Object.assign(rig.fit, S.fit); }
  tracker.wantHands = S.hands; tracker.wantMask = S.mask;
  scene.shared.uMaskOn.value = S.mask && maskReady ? 1 : 0;
  scene.shared.uMaskLo.value = 0.5 - 0.42 * S.feather; scene.shared.uMaskHi.value = 0.55 + 0.25 * S.feather;
  scene.shared.uFeather.value = S.feather;
  $("readout").hidden = !S.readout;
  if (scene.loaded) scene.applyLook(S);
  if (coach && !S.coachMuscles) scene.setActivation({});
  save();
}

function syncUI() {
  for (const [id, [get, , fmt]] of Object.entries(SLIDERS)) { $(id).value = get(); if (fmt && $(id + "V")) $(id + "V").textContent = fmt(get()); }
  for (const [id, key] of Object.entries(SWITCHES)) $(id).checked = !!S[key];
  $("reveal").value = S.reveal; $("model").value = S.model; $("resSel").value = S.res;
  $("cutRow").hidden = S.reveal === "whole"; $("cutLabel").textContent = S.reveal === "lens" ? "Lens size" : "Cut position";
  for (const b of $("lookSeg").children) b.classList.toggle("on", b.dataset.look === S.look);
  $("btnMirror").classList.toggle("on", S.mirror);
}

// ----------------------------------------------------------------------------- sizing
let W = 1280, H = 720, dpr = 1;
function resize() {
  const r = stage.getBoundingClientRect();
  W = Math.max(2, Math.round(r.width)); H = Math.max(2, Math.round(r.height)); dpr = Math.min(window.devicePixelRatio || 1, 2);
  scene.resize(W, H, dpr); hud.resize(W, H, dpr);
  const bw = $("badge").getBoundingClientRect(); badge.resize(Math.max(2, bw.width), Math.max(2, bw.height), dpr);
}
new ResizeObserver(resize).observe(stage);
function setAspect() { if (video.videoWidth && video.videoHeight) stage.style.setProperty("--ar", (video.videoWidth / video.videoHeight).toFixed(4)); }

// ----------------------------------------------------------------------------- start-up
let booted = null;
async function boot() {
  if (booted) return booted;
  booted = (async () => {
    loader("Loading the 3D anatomy…", 0.05);
    rig = null;
    try { await badge.load("assets/fonts/droid_sans_bold.typeface.json"); resize(); }
    catch (e) { $("badge").parentElement.classList.add("flat"); console.warn("badge", e); }
    const man = await scene.load("assets/anatomy/", null, null, (f, name) => loader("Loading 3D anatomy: " + name + "…", 0.05 + f * 0.5));
    rig = new Rig(man);
    scene.segData = rig.segData; scene.shared.uSegA.value = rig.alpha;
    labels = man.labels;
    for (const L of man.layers) counts[L.name] = (L.parts.length > 1 ? L.parts.length + " parts · " : "") + Math.round(L.nt / 1000) + "k polys";
    buildLayerList();
    rig.setSmoothing(S.smooth); afterChange();
    coach = new Coach({ scene, toast, settings: S, onChange: afterChange });
    await tracker.init(S.model, txt => loader(txt, 0.7));
    updateEngineChip();
    resize();
    loader(null);
  })();
  try { await booted; } catch (e) { booted = null; throw e; }
  return booted;
}
function updateEngineChip() { $("chipEngine").textContent = (tracker.poseModel || "–") + " model · " + (tracker.delegate === "GPU" ? "GPU" : "CPU"); }

async function startCamera() {
  try {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia)
      throw Object.assign(new Error("This browser does not allow camera access here. Open the app through the start file (http://localhost) in Chrome or Edge."), { name: "NoMedia" });
    stopSource();
    loader("Asking for the camera…", 0.02);
    const wide = parseInt(S.res, 10), tall = Math.round(wide * 9 / 16);
    const v = { width: { ideal: wide }, height: { ideal: wide === 640 ? 480 : tall }, frameRate: { ideal: 30, max: 60 } };
    if (S.camera) v.deviceId = { exact: S.camera }; else v.facingMode = "user";
    stream = await navigator.mediaDevices.getUserMedia({ video: v, audio: false });
    for (const tr of stream.getVideoTracks()) tr.onended = () => { toast("The camera was disconnected. Click Start camera to reconnect.", true, 6000); stopSource(true); };
    video.srcObject = stream; video.loop = false;
    await video.play();
    source = "camera"; S.mirror = S.mirror ?? true;
    listCameras();
    await afterSource();
  } catch (err) {
    loader(null);
    const denied = err && (err.name === "NotAllowedError" || err.name === "SecurityError");
    const busy = err && err.name === "NotReadableError";
    toast(denied ? "Camera blocked. Click the camera icon in the address bar, choose Allow, then Start camera again."
        : busy ? "The camera is being used by another app (Zoom, Teams, Camera…). Close it and try again."
        : "Could not start: " + (err.message || err), true, 8000);
    $("intro").hidden = false;
  }
}

async function startFile(url, name) {
  stopSource();
  video.srcObject = null; video.src = url; video.loop = true; video.muted = true;
  await new Promise((ok, bad) => { video.onloadeddata = ok; video.onerror = () => bad(new Error("this video format cannot be played")); });
  await video.play();
  source = "file"; if (name) toast("Playing " + name);
  await afterSource();
}

async function afterSource() {
  setAspect(); $("intro").hidden = true;
  scene.setVideo(video); scene.setMirror(S.mirror);
  try { await boot(); } catch (err) { loader(null); toast("Could not load: " + err.message, true, 9000); $("intro").hidden = false; return; }
  loader(null); resize(); running = true; setChip("warn", "Looking for you…");
}

function stopSource(showIntro = false) {
  running = false;
  if (stream) { for (const t of stream.getTracks()) t.stop(); stream = null; }
  if (video.src && video.src.startsWith("blob:")) URL.revokeObjectURL(video.src);
  video.pause(); video.removeAttribute("src"); video.srcObject = null; source = null;
  setChip("", "Idle"); if (showIntro) $("intro").hidden = false;
}

async function listCameras() {
  try {
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "videoinput");
    const sel = $("camSel"); sel.innerHTML = '<option value="">Default camera</option>';
    devs.forEach((d, i) => { const o = document.createElement("option"); o.value = d.deviceId; o.textContent = d.label || "Camera " + (i + 1); sel.appendChild(o); });
    sel.value = S.camera || "";
  } catch (e) {}
}

$("btnStart").onclick = () => startCamera();
$("fileInput").onchange = e => { const f = e.target.files[0]; if (f) startFile(URL.createObjectURL(f), f.name).catch(err => toast("Could not open the video: " + err.message, true)); e.target.value = ""; };
$("btnStop").onclick = () => stopSource(true);
$("btnMirror").onclick = () => { S.mirror = !S.mirror; scene.setMirror(S.mirror); syncUI(); save(); };
$("btnFull").onclick = () => { if (document.fullscreenElement) document.exitFullscreen(); else stage.requestFullscreen().catch(() => {}); };
$("btnSnap").onclick = snapshot;
$("btnRec").onclick = toggleRecord;
$("btnCalib").onclick = () => {
  if (!rig || !rig.found) { toast("Stand in front of the camera first.", true); return; }
  rig.resetCalibration(); $("calibInfo").textContent = "Measuring… stand still, arms slightly away from your body.";
  setTimeout(() => { const ok = rig.calibrateFromWorld(); $("calibInfo").textContent = ok ? "Calibrated to your body. Bone lengths are locked." : "Could not see all limbs clearly. Step back so your whole body is visible and try again."; toast(ok ? "Calibration done" : "Calibration incomplete", !ok); }, 3000);
};
window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
  if (e.key === "f" || e.key === "F") $("btnFull").click();
  if (e.key === "p" || e.key === "P") snapshot();
  if (e.key === "r" || e.key === "R") toggleRecord();
  if (e.key === "m" || e.key === "M") $("btnMirror").click();
  const n = parseInt(e.key, 10); if (n >= 1 && n <= LAYERS.length) { const L = LAYERS[n - 1]; S.layers[L.id] = !S.layers[L.id]; buildLayerList(); afterChange(); }
});
// WebGL can be reset by the graphics driver; recover instead of freezing
$("gl").addEventListener("webglcontextlost", e => { e.preventDefault(); toast("Graphics were reset by the system. Reloading…", true); setTimeout(() => location.reload(), 1500); });

// ----------------------------------------------------------------------------- per-frame
let lastVideoTime = -1, lastT = performance.now(), frames = 0, fps = 0, fpsAt = performance.now(), procFrames = 0, procFps = 0;
let maskReady = false, consecutiveErrors = 0, pointer = null, lensPos = null, readoutAt = 0, metricsAt = 0, lastMetrics = {};
stage.addEventListener("pointermove", e => { const r = stage.getBoundingClientRect(); pointer = { x: e.clientX - r.left, y: e.clientY - r.top }; });
stage.addEventListener("pointerleave", () => (pointer = null));

let stepMode = false;                                       // test mode: frames are fed one by one
function process(now, force = false) {
  if (!running || video.readyState < 2 || (document.hidden && !force)) return;
  if (video.currentTime === lastVideoTime && !force) return;  // only new camera frames are tracked
  lastVideoTime = video.currentTime;
  let res = null;
  try { res = tracker.detect(video); consecutiveErrors = 0; }
  catch (err) {
    console.error(err);
    if (++consecutiveErrors === 5) {
      toast("Tracking hit a problem, restarting it on the processor…", true);
      tracker.delegate = "CPU"; tracker.close(); running = false;
      tracker.init(S.model, t => loader(t, 0.7)).then(() => { loader(null); running = true; updateEngineChip(); }).catch(e => toast("Tracking failed: " + e.message, true, 9000));
    }
    return;
  }
  const t = now / 1000;
  if (res && res.mask && S.mask) { scene.setMask(res.mask, res.maskW, res.maskH); if (!maskReady) { maskReady = true; afterChange(); } }
  rig.update(res, W, H, t);
  if (coach) coach.frame(res, now);
  procFrames++;
  if (rig.found) {
    const m = measure(rig); lastMetrics = m;
    if (log.recording) log.add(now, rig, m, fps, tracker.inferMs);
    log.push(now, m);
  }
}

function updateLens(t) {
  if (S.reveal !== "lens" || !rig) { scene.shared.uLensOn.value = 0; return null; }
  let target = null;
  for (const s of ["R", "L"]) if (!target && rig.handsTracked[s]) { const p = rig.H[s][8]; target = { x: S.mirror ? W - p.x : p.x, y: H - p.y }; }
  if (!target && pointer) target = pointer;
  if (!target && rig.found) { const p = rig.P[0]; target = { x: S.mirror ? W - p.x : p.x, y: H - p.y }; }
  if (!target) target = { x: W / 2, y: H / 2 };
  if (!lensPos) lensPos = { ...target }; else { lensPos.x += (target.x - lensPos.x) * 0.35; lensPos.y += (target.y - lensPos.y) * 0.35; }
  const r = (0.08 + 0.32 * S.cut) * Math.min(W, H) * 1.4;
  scene.shared.uLensOn.value = 1; scene.shared.uLens.value.set(lensPos.x * dpr, (H - lensPos.y) * dpr, r * dpr);
  return { x: lensPos.x, y: lensPos.y, r };
}
function updateReveal() {
  const u = scene.shared; u.uClipOn.value = 1;
  const c = S.cut;
  switch (S.reveal) {
    case "left": u.uClip.value.set(1, 0, 0, (c - 0.5) * 0.4); break;
    case "right": u.uClip.value.set(-1, 0, 0, (c - 0.5) * 0.4); break;
    case "upper": u.uClip.value.set(0, 1, 0, 0.55 + (1 - c) * 0.95); break;
    case "lower": u.uClip.value.set(0, -1, 0, -(0.55 + c * 0.95)); break;
    default: u.uClipOn.value = 0;
  }
  u.uClipSoft.value = 0.005 + S.feather * 0.05;
}

function loop(now) {
  requestAnimationFrame(loop);
  const dt = Math.min(0.1, (now - lastT) / 1000); lastT = now;
  if (stepMode) { badge.render(); return; }
  if (running && rig) process(now);
  if (rig) rig.tick(dt, now / 1000);
  if (scene.loaded && rig) {
    updateReveal(); const lens = updateLens(now / 1000);
    scene.render(now / 1000);
    hud.draw(rig, labels, S, S.mirror, lens);
    if (recorder) composite(recCtx);
  } else if (source) scene.renderer.render(scene.scene, scene.cam);
  badge.render();
  frames++;
  if (now - fpsAt >= 1000) {
    fps = Math.round(frames * 1000 / (now - fpsAt)); procFps = Math.round(procFrames * 1000 / (now - fpsAt)); frames = procFrames = 0; fpsAt = now;
    $("chipFps").textContent = procFps + " fps · " + tracker.inferMs.toFixed(0) + " ms";
  }
  if (running && rig && now - metricsAt > 200) { metricsAt = now; updateStatus(); updateDataPane(); }
  if (S.readout && rig && now - readoutAt > 150) { readoutAt = now; updateReadout(); }
}
requestAnimationFrame(loop);

function updateStatus() {
  const q = rig.quality, found = rig.found;
  const hands = (rig.handsTracked.L ? 1 : 0) + (rig.handsTracked.R ? 1 : 0);
  if (!found) setChip("bad", "No person in view");
  else if (q < 0.55) setChip("warn", "Weak tracking · more light / face the camera");
  else setChip("ok", (rig.vis[23] > 0.5 && rig.vis[25] > 0.5 ? "Tracking full body" : "Tracking upper body") + (hands ? " + " + hands + (hands > 1 ? " hands" : " hand") : ""));
  $("chipQuality").textContent = "quality " + (found ? Math.round(q * 100) + "%" : "–");
}

function updateReadout() {
  const w = rig.world, m = lastMetrics, f = v => v === null || v === undefined ? "  –" : v.toFixed(0).padStart(4) + "°";
  $("readout").textContent = [
    "SEE YOUR LIFE X-RAY  ·  ENGINE",
    "model      " + tracker.poseModel + " / " + tracker.delegate,
    "tracking   " + procFps + " fps, " + tracker.inferMs.toFixed(1) + " ms",
    "quality    " + Math.round(rig.quality * 100) + "%",
    "scale      " + (rig.pxPerM.v || 0).toFixed(0) + " px/m",
    "bones      " + (rig.calibrated ? "locked (calibrated)" : "learning…"),
    "hands      " + ["L", "R"].filter(s => rig.handsTracked[s]).join(" + ") || "none",
    "elbow L/R  " + f(m.l_elbow) + " " + f(m.r_elbow),
    "knee  L/R  " + f(m.l_knee) + " " + f(m.r_knee),
  ].join("\n");
}

// ----------------------------------------------------------------------------- data pane
const chartSel = $("chartSel");
for (const m of METRICS) { const o = document.createElement("option"); o.value = m.key; o.textContent = m.name; chartSel.appendChild(o); }
chartSel.value = "r_elbow";
function updateDataPane() {
  const q = rig.found ? rig.quality : 0;
  $("qBar").style.width = Math.round(q * 100) + "%"; $("qBar").style.background = q > 0.75 ? "var(--ok)" : q > 0.5 ? "var(--amber)" : "var(--bad)";
  $("qText").textContent = rig.found ? Math.round(q * 100) + "%" : "no person";
  if (!document.querySelector('[data-pane="data"]').classList.contains("on")) return;
  const rows = METRICS.map(m => { const v = lastMetrics[m.key]; return `<tr><td>${m.name}</td><td class="${v == null ? "na" : ""}">${v == null ? "not visible" : v.toFixed(m.dp ?? 0) + m.unit}</td></tr>`; });
  $("metrics").innerHTML = rows.join("");
  drawChart($("chart"), log.series(chartSel.value), METRICS.find(m => m.key === chartSel.value));
}
$("btnDataRec").onclick = () => {
  if (!log.recording) { log.start(); $("btnDataRec").textContent = "■ Stop data recording"; $("btnCsv").disabled = $("btnReport").disabled = true; $("dataInfo").textContent = "Recording… move naturally."; }
  else { log.stop(); $("btnDataRec").textContent = "● Start data recording"; $("btnCsv").disabled = $("btnReport").disabled = !log.rows.length;
         $("dataInfo").textContent = log.rows.length + " frames recorded over " + log.duration().toFixed(1) + " s."; }
};
$("btnCsv").onclick = () => download(new Blob([log.csv()], { type: "text/csv" }), "see-your-life-xray_data_" + stamp() + ".csv");
$("btnReport").onclick = () => download(new Blob([JSON.stringify(log.report({ model: tracker.poseModel, delegate: tracker.delegate }), null, 2)], { type: "application/json" }), "see-your-life-xray_report_" + stamp() + ".json");

// ----------------------------------------------------------------------------- photo and video
const stamp = () => new Date().toISOString().replace(/[:T]/g, "-").slice(0, 19);
function download(blob, name) { const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(a.href), 5000); }
function composite(ctx) {
  const c = ctx.canvas; ctx.drawImage($("gl"), 0, 0, c.width, c.height); ctx.drawImage($("hud"), 0, 0, c.width, c.height);
}
function snapshot() {
  if (!scene.loaded) return;
  const c = document.createElement("canvas"); c.width = $("gl").width; c.height = $("gl").height;
  scene.render(performance.now() / 1000); composite(c.getContext("2d"));
  c.toBlob(b => { download(b, "see-your-life-xray_" + stamp() + ".png"); toast("Picture saved to your Downloads folder"); }, "image/png");
}
let recCtx = null, recChunks = [];
function toggleRecord() {
  if (recorder) { recorder.stop(); return; }
  if (!scene.loaded || !source) { toast("Start the camera first.", true); return; }
  const c = document.createElement("canvas"); c.width = $("gl").width; c.height = $("gl").height; recCtx = c.getContext("2d");
  const type = ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm", "video/mp4"].find(t => window.MediaRecorder && MediaRecorder.isTypeSupported(t));
  if (!type) { toast("Video recording is not supported in this browser.", true); return; }
  recChunks = []; recorder = new MediaRecorder(c.captureStream(30), { mimeType: type, videoBitsPerSecond: 8e6 });
  recorder.ondataavailable = e => e.data.size && recChunks.push(e.data);
  recorder.onstop = () => { download(new Blob(recChunks, { type }), "see-your-life-xray_" + stamp() + (type.includes("mp4") ? ".mp4" : ".webm")); recorder = null; recCtx = null;
                            $("btnRec").classList.remove("rec"); $("btnRec").querySelector("span").textContent = "Record"; toast("Video saved to your Downloads folder"); };
  recorder.start(500); $("btnRec").classList.add("rec"); $("btnRec").querySelector("span").textContent = "Stop recording";
}

// ----------------------------------------------------------------------------- init
syncUI(); buildLayerList(); afterChange();
boot().catch(err => { loader(null); toast("Could not load the app files: " + err.message + ". Start the app with the start file, not by double-clicking index.html.", true, 12000); });
if (params.get("video"))       // ?video=path plays a file from the app folder (loaded fully first so it can be searched)
  fetch(params.get("video")).then(r => { if (!r.ok) throw new Error("video not found"); return r.blob(); })
    .then(b => startFile(URL.createObjectURL(b), null)).catch(e => toast(e.message, true));

if (TEST) window.__slx = {
  // feed one video frame with a simulated clock (used by the automated tests)
  step(simMs, dt = 1 / 30) { stepMode = true; process(simMs, true); rig.tick(dt, simMs / 1000); updateReveal(); const lens = updateLens(simMs / 1000);
    scene.render(simMs / 1000); hud.draw(rig, labels, S, S.mirror, lens); updateStatus(); return rig.found; },
  apply() { buildLayerList(); syncUI(); afterChange(); }, S, scene, tracker, get rig() { return rig; }, get fps() { return procFps; }, get metrics() { return lastMetrics; }, get coach() { return coach; }, log, NSEG };

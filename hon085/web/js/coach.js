// See your Life X-ray - pose coach client: talks to the Python server and fills the Coach and Train tabs
const $ = id => document.getElementById(id);
const SEND_INTERVAL_MS = 80;
const TEACH_SECONDS = 5, TEACH_COUNTDOWN = 3;
const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24], [23, 24], [23, 25], [25, 27], [24, 26], [26, 28], [27, 31], [28, 32], [27, 29], [28, 30]];

async function api(path, options) {
  const r = await fetch(path, options);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
}
const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const flat = lm => lm.map(p => [+p.x.toFixed(4), +p.y.toFixed(4), +p.z.toFixed(4)]);
const vis = lm => lm.map(p => +(p.visibility ?? 0).toFixed(3));

export class Coach {
  constructor({ scene, toast, settings, onChange }) {
    this.scene = scene; this.toast = toast; this.S = settings; this.onChange = onChange;
    this.socket = null; this.lastSent = 0; this.model = null; this.reply = null; this.teach = null; this.online = false;
    $("coachOn").onchange = e => { this.S.coach = e.target.checked; this.onChange(); if (!this.S.coach) this.clear(); };
    $("btnSummary").onclick = () => this.send({ type: "summary" });
    $("btnCoachReset").onclick = () => { this.send({ type: "reset" }); $("sessionTable").innerHTML = ""; this.clear(); };
    $("photoInput").onchange = e => { const f = e.target.files[0]; if (f) this.analysePhoto(f); e.target.value = ""; };
    $("btnTeach").onclick = () => this.startTeaching();
    $("btnTrain").onclick = () => this.train();
    this.refreshModel();
    this.connect();
  }

  connect() {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/coach`);
    ws.onopen = () => { this.online = true; };
    ws.onmessage = e => this.receive(JSON.parse(e.data));
    ws.onclose = () => { this.online = false; this.socket = null; setTimeout(() => this.connect(), 2000); };
    ws.onerror = () => ws.close();
    this.socket = ws;
  }
  send(msg) { if (this.socket && this.socket.readyState === 1) this.socket.send(JSON.stringify(msg)); }

  // called for every tracked camera frame
  frame(res, now) {
    if (!res || !res.world) return;
    if (this.teach && this.teach.recording) { this.teach.world.push(flat(res.world)); this.teach.visibility.push(vis(res.world)); }
    if (!this.S.coach || now - this.lastSent < SEND_INTERVAL_MS) return;
    this.lastSent = now;
    this.send({ t: Math.round(now), world: flat(res.world), visibility: vis(res.world) });
  }

  receive(msg) {
    if (msg.type === "summary") return this.showSummary(msg);
    if (msg.type !== "coach" || !this.S.coach) return;
    this.reply = msg;
    this.scene.setActivation(this.S.coachMuscles ? msg.muscles || {} : {});
    this.render(msg);
  }

  clear() { this.reply = null; this.scene.setActivation({}); this.render({ status: "off" }); }

  render(m) {
    const known = !!m.known, banner = $("poseBanner");
    const name = m.status === "no_model" ? "Model not trained" : m.status === "low_visibility" ? "Step back, whole body in view"
      : m.status === "off" ? "Coach is off" : known ? m.title : "No known pose";
    $("poseName").textContent = name; $("poseCue").textContent = known ? m.cue : "";
    $("poseCard").classList.toggle("known", known);
    $("poseScore").textContent = known && m.score != null ? m.score : "–";
    $("poseConf").textContent = known ? Math.round(m.confidence * 100) + "% sure" : "";
    $("poseHold").textContent = known ? m.hold_seconds.toFixed(1) + " s hold" : "";
    $("poseReps").textContent = known ? "× " + m.repetitions : "";
    $("poseHints").innerHTML = known ? (m.hints.length ? m.hints.map(h => `<li>${esc(h)}</li>`).join("") : '<li class="good">Good form, keep holding</li>') : "";
    const chips = known ? [...m.primary.map(t => `<span class="mchip hot">${esc(t)}</span>`), ...m.secondary.map(t => `<span class="mchip">${esc(t)}</span>`)] : [];
    $("muscleChips").innerHTML = chips.join("") || '<span class="hint">Hold one of the poses the model knows.</span>';
    const probs = m.probabilities || {};
    $("probBars").innerHTML = Object.entries(probs).sort((a, b) => b[1] - a[1]).map(([c, p]) =>
      `<div class="prob${c === m.label ? " top" : ""}"><span>${esc(c)}</span><div><i style="width:${Math.round(p * 100)}%"></i></div><b>${Math.round(p * 100)}%</b></div>`).join("");
    banner.hidden = !known;
    if (known) banner.innerHTML = `<b>${esc(m.title)}</b><span>${Math.round(m.confidence * 100)}% · ${m.hold_seconds.toFixed(0)} s${m.score != null ? " · form " + m.score : ""}</span>`;
  }

  showSummary(msg) {
    const rows = msg.poses.map(p => `<tr><td>${esc(p.title)}</td><td>${p.seconds} s · ×${p.times_entered} · best ${p.longest_hold} s${p.mean_score != null ? " · form " + p.mean_score : ""}</td></tr>`);
    $("sessionTable").innerHTML = rows.join("") || '<tr><td>No pose held yet</td><td></td></tr>';
  }

  async refreshModel() {
    try { this.model = await api("/api/model"); } catch (e) { this.model = null; $("modelTable").innerHTML = '<tr><td>Python server not reachable</td><td></td></tr>'; return; }
    const m = this.model;
    this.scene.setMuscleGroups(m.muscle_groups);
    const rows = [];
    if (m.trained) {
      const i = m.info, v = i.validation, e = m.evaluation;
      rows.push(["Algorithm", i.algorithm_name], ["Poses", m.classes.map(c => c.label).join(", ")], ["Training poses", i.n_samples + (i.user_samples ? ` (${i.user_samples} yours)` : "")],
        ["Features", i.n_features + " per pose"]);
      if (v) rows.push(["Validation", `${(v.accuracy * 100).toFixed(1)}% (${v.method})`]);
      if (e) rows.push(["Held-out test", `${(e.test_accuracy * 100).toFixed(1)}% accuracy, F1 ${e.test_macro_f1.toFixed(3)}`]);
      rows.push(["Trained", i.trained_at]);
    } else rows.push(["Status", "Not trained yet. Run python scripts/train.py or teach poses below."]);
    $("modelTable").innerHTML = rows.map(([k, v]) => `<tr><td>${esc(k)}</td><td>${esc(v)}</td></tr>`).join("");
    const sel = $("algoSel"), current = m.trained ? m.info.algorithm : "mlp";
    sel.innerHTML = Object.entries(m.algorithms).map(([k, n]) => `<option value="${k}">${esc(n)}</option>`).join(""); sel.value = current;
    if (!$("teachMuscles").children.length)
      $("teachMuscles").innerHTML = m.muscle_groups.map(g => `<label class="pick"><input type="checkbox" value="${g.id}"><span>${esc(g.title)}</span></label>`).join("");
    this.showSamples(m.user_samples);
  }

  showSamples(counts) {
    const rows = Object.entries(counts || {}).map(([label, n]) => `<tr><td>${esc(label)}</td><td>${n} samples <button class="linkbtn del" data-label="${esc(label)}">delete</button></td></tr>`);
    $("sampleTable").innerHTML = rows.join("");
    for (const b of $("sampleTable").querySelectorAll(".del")) b.onclick = async () => {
      try { const r = await api("/api/samples/" + encodeURIComponent(b.dataset.label), { method: "DELETE" }); this.showSamples(r.counts); this.toast("Samples deleted. Train again to update the model."); }
      catch (e) { this.toast(e.message, true); }
    };
  }

  startTeaching() {
    const label = $("teachName").value.trim();
    if (!label) { this.toast("Type a name for the pose first.", true); return; }
    if (this.teach) return;
    const info = $("teachInfo"), t = this.teach = { label, world: [], visibility: [], recording: false };
    let left = TEACH_COUNTDOWN;
    const tick = () => {
      if (left > 0) { info.textContent = `Get into the pose… ${left}`; left--; setTimeout(tick, 1000); return; }
      t.recording = true; info.textContent = "Recording… hold the pose and move a little.";
      setTimeout(() => this.finishTeaching(), TEACH_SECONDS * 1000);
    };
    tick();
  }

  async finishTeaching() {
    const t = this.teach; this.teach = null; t.recording = false;
    const info = $("teachInfo");
    if (!t.world.length) { info.textContent = "Nothing recorded. Start the camera and stand in view."; return; }
    const primary = [...$("teachMuscles").querySelectorAll("input:checked")].map(i => i.value);
    try {
      const r = await api("/api/samples", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ label: t.label, world: t.world, visibility: t.visibility, primary }) });
      info.textContent = `${r.stored} samples saved for "${r.label}". Record more or press Train.`;
      this.showSamples(r.counts);
    } catch (e) { info.textContent = "Could not save: " + e.message; }
  }

  async train() {
    const btn = $("btnTrain"), info = $("trainInfo");
    btn.disabled = true; info.textContent = "Training…";
    try {
      this.model = await api("/api/train", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ algorithm: $("algoSel").value }) });
      const v = this.model.info.validation;
      info.textContent = `Done. ${this.model.classes.length} poses` + (v ? `, validation accuracy ${(v.accuracy * 100).toFixed(1)}%.` : ".");
      await this.refreshModel();
    } catch (e) { info.textContent = "Training failed: " + e.message; }
    btn.disabled = false;
  }

  async analysePhoto(file) {
    const box = $("photoResult"), text = $("photoText"), canvas = $("photoCanvas");
    box.hidden = false; text.textContent = "Analysing on the Python server…";
    const form = new FormData(); form.append("file", file);
    let r;
    try { r = await api("/api/analyze", { method: "POST", body: form }); } catch (e) { text.textContent = "Could not analyse: " + e.message; return; }
    const img = await createImageBitmap(file).catch(() => null);
    if (img) {
      const w = canvas.width = 640, h = canvas.height = Math.round(640 * img.height / img.width), g = canvas.getContext("2d");
      g.drawImage(img, 0, 0, w, h);
      if (r.found) {
        g.lineWidth = 3; g.strokeStyle = "#5eead4"; g.fillStyle = "#fff";
        for (const [a, b] of BONES) { g.beginPath(); g.moveTo(r.landmarks[a][0] * w, r.landmarks[a][1] * h); g.lineTo(r.landmarks[b][0] * w, r.landmarks[b][1] * h); g.stroke(); }
        for (const i of new Set(BONES.flat())) { g.beginPath(); g.arc(r.landmarks[i][0] * w, r.landmarks[i][1] * h, 4, 0, 6.3); g.fill(); }
      }
    }
    if (!r.found) { text.textContent = "No person was found in this picture."; return; }
    const top = Object.entries(r.probabilities).sort((a, b) => b[1] - a[1]).slice(0, 3).map(([c, p]) => `${esc(c)} ${Math.round(p * 100)}%`).join(" · ");
    text.innerHTML = (r.known ? `<b>${esc(r.title)}</b> (${Math.round(r.confidence * 100)}% sure)` : `<b>No known pose</b> (closest: ${esc(r.closest)})`) +
      `<br><span class="hint">${top}</span>` + (r.known && r.primary.length ? `<br>Main muscles: ${esc(r.primary.join(", "))}` : "") +
      `<br><span class="hint">Knees ${r.angles.left_knee}° / ${r.angles.right_knee}° · elbows ${r.angles.left_elbow}° / ${r.angles.right_elbow}°</span>`;
  }
}

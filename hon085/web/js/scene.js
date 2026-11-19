// See your Life X-ray - 3D scene: camera picture, body mask and the rigged anatomy layers
import * as THREE from "three";
import { NSEG } from "./rig.js";

const MAX_GROUPS = 24;

export const LAYERS = [
  { id: "skin",     name: "Body outline",      color: "#5cc8ff", xray: 0x5cc8ff, gain: 0.5 },
  { id: "skeleton", name: "Skeleton",          color: "#e9f3ff", xray: 0xdcecff, gain: 0.62 },
  { id: "muscles",  name: "Muscles",           color: "#ff5a4e", xray: 0xff6a5a, gain: 0.42 },
  { id: "heart",    name: "Heart",             color: "#ff2d55", xray: 0xff3b5c, gain: 1.0 },
  { id: "vessels",  name: "Blood vessels",     color: "#ff7a45", xray: 0xff5f3d, gain: 0.85 },
  { id: "lungs",    name: "Lungs & airways",   color: "#ff9ec4", xray: 0xff9ec4, gain: 0.55 },
  { id: "organs",   name: "Digestive & kidneys", color: "#ffc061", xray: 0xffb35c, gain: 0.55 },
  { id: "brain",    name: "Brain & eyes",      color: "#c69bff", xray: 0xc8a2ff, gain: 0.6 },
];

const VERT = /* glsl */`
attribute vec2 seg; attribute float sw; attribute vec3 col; attribute float grp;
uniform mat4 uSeg[${NSEG}]; uniform float uSegA[${NSEG}]; uniform float uAct[${MAX_GROUPS}];
uniform float uPulse; uniform vec3 uPulseC; uniform float uTime;
varying vec3 vN; varying vec3 vCol; varying vec3 vRest; varying float vA; varying float vAct;
void main() {
  vAct = grp < -0.5 ? 0.0 : uAct[int(grp + 0.5)];
  int i0 = int(seg.x + 0.5), i1 = int(seg.y + 0.5);
  vec3 p0 = position;
  if (uPulse > 0.0) {                                   // illustrative heartbeat (not measured)
    float ph = fract(uTime * 1.2);
    float beat = exp(-pow((ph - 0.08) * 14.0, 2.0)) + 0.55 * exp(-pow((ph - 0.3) * 14.0, 2.0));
    p0 = uPulseC + (p0 - uPulseC) * (1.0 + uPulse * beat);
  }
  mat4 M = uSeg[i0] * sw + uSeg[i1] * (1.0 - sw);
  vec4 wp = M * vec4(p0, 1.0);
  vN = normalize(mat3(M) * normal);
  vCol = col; vRest = position;
  vA = uSegA[i0] * sw + uSegA[i1] * (1.0 - sw);
  gl_Position = projectionMatrix * viewMatrix * wp;
}`;

const FRAG = /* glsl */`
uniform float uMode, uGain, uOpacity, uTime, uMaskOn, uMaskLo, uMaskHi, uMirror, uFeather, uRim;
uniform vec3 uTint; uniform sampler2D uMask; uniform vec2 uRes;
uniform vec4 uClip; uniform float uClipOn, uClipSoft; uniform vec3 uLens; uniform float uLensOn;
varying vec3 vN; varying vec3 vCol; varying vec3 vRest; varying float vA; varying float vAct;
const vec3 HOT = vec3(1.0, 0.78, 0.2);
void main() {
  float a = vA * uOpacity;
  if (a < 0.003) discard;
  vec2 sc = gl_FragCoord.xy / uRes;
  if (uMaskOn > 0.5) {                               // keep the anatomy inside your silhouette
    vec2 iuv = vec2(uMirror > 0.5 ? 1.0 - sc.x : sc.x, 1.0 - sc.y);
    a *= smoothstep(uMaskLo, uMaskHi, texture2D(uMask, iuv).r);
  }
  if (uClipOn > 0.5) {                               // reveal: half body, upper/lower body
    float d = dot(uClip.xyz, vRest) - uClip.w;
    if (d < 0.0) discard;
    a *= smoothstep(0.0, uClipSoft, d);
  }
  if (uLensOn > 0.5) {                               // x-ray lens around the fingertip / pointer
    float dl = length(gl_FragCoord.xy - uLens.xy);
    a *= 1.0 - smoothstep(uLens.z * (1.0 - 0.35 * uFeather), uLens.z, dl);
  }
  if (a < 0.003) discard;
  vec3 n = normalize(vN);
  float ndv = abs(n.z);
  if (uMode < 0.5) {                                 // X-ray: glowing edges, see-through faces
    float rim = pow(1.0 - ndv, uRim);
    float k = clamp((0.05 + 0.95 * rim) * uGain, 0.0, 1.0);
    vec3 c = mix(uTint, vCol, 0.18) * (0.75 + 0.6 * rim);
    float glow = vAct * (0.75 + 0.25 * sin(uTime * 5.0));
    c = mix(c, HOT, glow); k = clamp(k * (1.0 + 0.5 * glow), 0.0, 1.0);
    gl_FragColor = vec4(c * k * a, 1.0);
  } else if (uMode < 1.5) {                          // Anatomical: lit, natural tissue colours
    vec3 fn = n.z < 0.0 ? -n : n;                  // always light the side that faces the camera
    vec3 L1 = normalize(vec3(0.35, 0.6, 0.72)), L2 = normalize(vec3(-0.6, 0.1, 0.5));
    float d = max(dot(fn, L1), 0.0) * 0.85 + max(dot(fn, L2), 0.0) * 0.3;
    vec3 h = normalize(L1 + vec3(0.0, 0.0, 1.0));
    float sp = pow(max(dot(fn, h), 0.0), 36.0) * 0.35;
    float rim = pow(1.0 - ndv, 3.0) * 0.35;
    vec3 c = vCol * (0.26 + d) * uGain + vec3(sp) + uTint * rim;
    c = mix(c, HOT * (0.45 + d), vAct * (0.7 + 0.2 * sin(uTime * 5.0)));
    gl_FragColor = vec4(c, a);
  } else {                                           // Hologram: scan lines and a moving scan band
    float rim = pow(1.0 - ndv, 1.6);
    float lines = 0.65 + 0.35 * sin(gl_FragCoord.y * 1.9);
    float band = exp(-pow((fract(uTime * 0.18) * 2.4 - 0.2) - (vRest.y / 1.75), 2.0) * 120.0);
    vec3 c = mix(mix(uTint, vec3(0.55, 1.0, 1.0), 0.35), HOT, vAct);
    float k = clamp((0.03 + 0.85 * rim) * uGain * lines * (0.8 + 2.2 * band) * (1.0 + 2.0 * vAct), 0.0, 1.2);
    gl_FragColor = vec4(c * k * a, 1.0);
  }
}`;

const BG_VERT = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0); }`;
const BG_FRAG = `
uniform sampler2D uVideo, uMask; uniform float uVeil, uMaskOn, uDim, uHasVideo; uniform vec3 uVeilColor;
varying vec2 vUv;
void main() {
  vec3 c = uHasVideo > 0.5 ? texture2D(uVideo, vUv).rgb : vec3(0.02, 0.03, 0.05);
  float l = dot(c, vec3(0.299, 0.587, 0.114));
  c = mix(c, vec3(l) * vec3(0.85, 0.95, 1.05), uDim * 0.6) * (1.0 - uDim * 0.55);    // calm the background
  if (uMaskOn > 0.5) {
    float m = smoothstep(0.2, 0.75, texture2D(uMask, vec2(vUv.x, 1.0 - vUv.y)).r);
    c = mix(c, uVeilColor + vec3(l) * 0.10, m * uVeil);       // darken your body so the anatomy reads clearly
  } else c = mix(c, uVeilColor + vec3(l) * 0.1, uVeil * 0.55);
  gl_FragColor = vec4(c, 1.0);
}`;

export class Scene {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: false, preserveDrawingBuffer: false, powerPreference: "high-performance" });
    this.renderer.setClearColor(0x05070c, 1); this.renderer.autoClear = true;
    this.scene = new THREE.Scene();
    this.cam = new THREE.OrthographicCamera(0, 1, 1, 0, -1e5, 1e5);
    this.W = 2; this.H = 2; this.mirror = true;
    this.maskData = null; this.maskTex = new THREE.DataTexture(new Uint8Array([0]), 1, 1, THREE.RedFormat); this.maskTex.needsUpdate = true;
    this.maskTex.magFilter = this.maskTex.minFilter = THREE.LinearFilter;
    this.videoTex = null; this.layers = {}; this.loaded = false;
    this.shared = {                                   // uniforms shared by every anatomy layer
      uSeg: { value: [] }, uSegA: { value: null }, uTime: { value: 0 }, uMask: { value: this.maskTex }, uRes: { value: new THREE.Vector2(1, 1) },
      uMaskOn: { value: 0 }, uMaskLo: { value: 0.12 }, uMaskHi: { value: 0.55 }, uMirror: { value: 1 }, uFeather: { value: 0.3 },
      uClip: { value: new THREE.Vector4(1, 0, 0, 0) }, uClipOn: { value: 0 }, uClipSoft: { value: 0.03 },
      uLens: { value: new THREE.Vector3(0, 0, 100) }, uLensOn: { value: 0 }, uAct: { value: new Float32Array(MAX_GROUPS) },
    };
    this.groupIds = []; this.actTarget = new Float32Array(MAX_GROUPS);
    this.bg = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.ShaderMaterial({
      uniforms: { uVideo: { value: null }, uMask: this.shared.uMask, uMaskOn: this.shared.uMaskOn, uVeil: { value: 0.75 },
                  uDim: { value: 0.25 }, uHasVideo: { value: 0 }, uVeilColor: { value: new THREE.Color(0x03070d) } },
      vertexShader: BG_VERT, fragmentShader: BG_FRAG, depthTest: false, depthWrite: false, side: THREE.DoubleSide }));
    this.bg.renderOrder = -100; this.bg.frustumCulled = false; this.scene.add(this.bg);
  }

  setVideo(video) {
    if (this.videoTex) this.videoTex.dispose();
    // VideoTexture, also refreshed by hand every frame (its own frame callback may not fire for an off-screen video)
    this.video = video; this.videoTex = new THREE.VideoTexture(video); this.videoTex.colorSpace = THREE.SRGBColorSpace;
    this.videoTex.minFilter = THREE.LinearFilter; this.videoTex.generateMipmaps = false;
    this.bg.material.uniforms.uVideo.value = this.videoTex; this.bg.material.uniforms.uHasVideo.value = 1;
  }

  resize(W, H, dpr) {
    this.W = W; this.H = H;
    this.renderer.setPixelRatio(dpr); this.renderer.setSize(W, H, false);
    this.shared.uRes.value.set(W * dpr, H * dpr);
    this.bg.scale.set(W, H, 1); this.bg.position.set(W / 2, H / 2, -5e4);
    this.setMirror(this.mirror);
  }
  setMirror(m) {
    this.mirror = m; this.shared.uMirror.value = m ? 1 : 0;
    if (m) { this.cam.left = this.W; this.cam.right = 0; } else { this.cam.left = 0; this.cam.right = this.W; }
    this.cam.top = this.H; this.cam.bottom = 0; this.cam.updateProjectionMatrix();
  }

  /** Copies the tracker's body mask (0..1 per pixel) into a small texture. */
  setMask(arr, w, h) {
    if (!arr) return;
    const step = w > 640 ? 2 : 1, mw = Math.floor(w / step), mh = Math.floor(h / step);
    if (!this.maskData || this.maskData.length !== mw * mh) {
      this.maskData = new Uint8Array(mw * mh);
      this.maskTex.dispose();
      this.maskTex = new THREE.DataTexture(this.maskData, mw, mh, THREE.RedFormat, THREE.UnsignedByteType);
      this.maskTex.magFilter = this.maskTex.minFilter = THREE.LinearFilter; this.maskTex.flipY = false;
      this.shared.uMask.value = this.maskTex;
    }
    const d = this.maskData;
    for (let y = 0, k = 0; y < mh; y++) { const row = y * step * w; for (let x = 0; x < mw; x++) d[k++] = Math.min(255, arr[row + x * step] * 255) | 0; }
    this.maskTex.needsUpdate = true;
  }

  async load(base, segData, segAlpha, onProgress = () => {}) {
    const layerBytes = L => L.offsets.idx + L.nt * 12;
    const man = await (await fetch(base + "manifest.json")).json();
    this.manifest = man;
    this.shared.uSegA.value = segAlpha;
    this.segMats = Array.from({ length: NSEG }, () => new THREE.Matrix4());
    this.shared.uSeg.value = this.segMats; this.segData = segData;
    let done = 0;
    for (const L of man.layers) {
      const res = await fetch(base + L.file);
      const buf = res.ok ? await res.arrayBuffer() : new ArrayBuffer(0);
      if (buf.byteLength < layerBytes(L)) throw new Error(L.file + " did not load. If a download manager (for example IDM) is installed, turn off its browser integration for localhost and reload");
      const o = L.offsets, nv = L.nv;
      const q = new Uint16Array(buf, o.pos, nv * 3), pos = new Float32Array(nv * 3);
      for (let i = 0; i < nv * 3; i++) pos[i] = L.min[i % 3] + q[i] / 65535 * L.size[i % 3];
      const g = new THREE.BufferGeometry();
      g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
      g.setAttribute("seg", new THREE.BufferAttribute(new Uint8Array(buf, o.seg, nv * 2), 2));
      g.setAttribute("sw", new THREE.BufferAttribute(new Uint8Array(buf, o.w, nv), 1, true));
      g.setAttribute("col", new THREE.BufferAttribute(new Uint8Array(buf, o.col, nv * 3), 3, true));
      g.setAttribute("grp", new THREE.BufferAttribute(new Float32Array(nv).fill(-1), 1));
      g.setIndex(new THREE.BufferAttribute(new Uint32Array(buf, o.idx, L.nt * 3), 1));
      g.computeVertexNormals();
      const meta = LAYERS.find(x => x.id === L.name);
      const heart = L.name === "heart";
      let pc = new THREE.Vector3();
      if (heart) { const p = L.parts.find(p => p.n === "wall of heart") || L.parts[0]; for (let i = p.v0; i < p.v0 + p.nv; i++) pc.add(new THREE.Vector3().fromArray(pos, i * 3)); pc.divideScalar(p.nv); }
      const mat = new THREE.ShaderMaterial({
        uniforms: { ...this.shared, uMode: { value: 0 }, uGain: { value: 1 }, uOpacity: { value: 1 }, uTint: { value: new THREE.Color(meta.xray) },
                    uRim: { value: 2.0 }, uPulse: { value: 0 }, uPulseC: { value: pc } },
        vertexShader: VERT, fragmentShader: FRAG, side: THREE.DoubleSide, transparent: true });
      const mesh = new THREE.Mesh(g, mat); mesh.frustumCulled = false; mesh.visible = false;
      mesh.renderOrder = LAYERS.indexOf(meta);
      this.scene.add(mesh); this.layers[L.name] = { mesh, meta, info: L };
      onProgress(++done / man.layers.length, meta.name);
    }
    this.loaded = true;
    return man;
  }

  /** Tags every muscle vertex with its muscle group so that groups can be lit up. groups: [{id, pattern}] */
  setMuscleGroups(groups) {
    const layer = this.layers.muscles;
    if (!layer || !groups) return;
    this.groupIds = groups.slice(0, MAX_GROUPS).map(g => g.id);
    const tests = groups.slice(0, MAX_GROUPS).map(g => new RegExp(g.pattern, "i"));
    const attr = layer.mesh.geometry.getAttribute("grp");
    for (const p of layer.info.parts) attr.array.fill(tests.findIndex(t => t.test(p.n)), p.v0, p.v0 + p.nv);
    attr.needsUpdate = true;
  }

  /** levels: {groupId: 0..1}; the glow fades towards these values. */
  setActivation(levels) {
    this.actTarget.fill(0);
    for (const [id, v] of Object.entries(levels)) { const i = this.groupIds.indexOf(id); if (i >= 0) this.actTarget[i] = v; }
  }

  /** look: 'xray' | 'anatomy' | 'holo' */
  applyLook(S) {
    const mode = S.look === "anatomy" ? 1 : S.look === "holo" ? 2 : 0;
    const on = LAYERS.filter(l => S.layers[l.id]);
    const heavy = on.filter(l => l.id !== "skin").length;
    const share = 1 / (0.45 + 0.55 * Math.max(1, heavy * 0.75));   // many glowing layers would wash out to white
    for (const id in this.layers) {
      const { mesh, meta } = this.layers[id], u = mesh.material.uniforms, m = mesh.material;
      mesh.visible = !!S.layers[id];
      const outline = id === "skin";
      const lm = outline ? (mode === 1 ? 0 : mode) : mode;          // the body outline always glows
      u.uMode.value = lm;
      if (lm === 1) { m.blending = THREE.NormalBlending; m.depthTest = true; m.depthWrite = true; u.uGain.value = 0.55 + 0.45 * S.bright; u.uOpacity.value = S.opacity; }
      else { m.blending = THREE.AdditiveBlending; m.depthTest = false; m.depthWrite = false;
             u.uGain.value = S.bright * meta.gain * (outline ? 1 : share) * (lm === 2 ? 1.1 : 1); u.uOpacity.value = 1; }
      u.uRim.value = outline ? 5.0 : 2.0;
      u.uPulse.value = id === "heart" && S.heartbeat ? 0.045 : 0;
      m.needsUpdate = true;
    }
    this.bg.material.uniforms.uVeil.value = 1 - S.real;
    this.bg.material.uniforms.uDim.value = S.dim;
  }

  render(t) {
    if (this.video && this.video.readyState >= 2) this.videoTex.needsUpdate = true;
    for (let i = 0; i < NSEG; i++) this.segMats[i].fromArray(this.segData, i * 16);
    this.shared.uTime.value = t;
    const act = this.shared.uAct.value;
    for (let i = 0; i < MAX_GROUPS; i++) act[i] += (this.actTarget[i] - act[i]) * 0.15;
    this.renderer.render(this.scene, this.cam);
  }
}

// See your Life X-ray - body and hand tracking (MediaPipe Tasks Vision, runs on this computer)
import { FilesetResolver, PoseLandmarker, HandLandmarker } from "../vendor/mediapipe/vision_bundle.mjs";

const WASM = new URL("../vendor/mediapipe/wasm", import.meta.url).href;
const MODEL = name => new URL("../models/" + name, import.meta.url).href;
export const POSE_MODELS = { lite: "pose_landmarker_lite.task", full: "pose_landmarker_full.task", heavy: "pose_landmarker_heavy.task" };

export class Tracker {
  constructor() {
    this.files = null; this.pose = null; this.hand = null; this.poseModel = null;
    this.delegate = "GPU"; this.lastTs = 0; this.errors = 0; this.wantMask = true; this.wantHands = true;
    this.inferMs = 0; this.handFrame = 0;
  }

  async init(poseModel = "full", onStep = () => {}) {
    onStep("Loading the tracking engine…");
    this.files = this.files || await FilesetResolver.forVisionTasks(WASM);
    await this.setPoseModel(poseModel, onStep);
    onStep("Loading finger tracking…");
    try { this.hand = await this.make(HandLandmarker, { baseOptions: { modelAssetPath: MODEL("hand_landmarker.task") },
      runningMode: "VIDEO", numHands: 2, minHandDetectionConfidence: 0.55, minHandPresenceConfidence: 0.55, minTrackingConfidence: 0.5 }); }
    catch (e) { console.warn("hand tracker unavailable", e); this.hand = null; }
  }

  async make(Cls, opts) {
    // Try the graphics card first (fast). Fall back to the processor if the GPU path is not available.
    try { return await Cls.createFromOptions(this.files, { ...opts, baseOptions: { ...opts.baseOptions, delegate: this.delegate } }); }
    catch (e) {
      if (this.delegate === "CPU") throw e;
      console.warn("GPU delegate failed, using CPU", e); this.delegate = "CPU";
      return await Cls.createFromOptions(this.files, { ...opts, baseOptions: { ...opts.baseOptions, delegate: "CPU" } });
    }
  }

  async setPoseModel(name, onStep = () => {}) {
    if (this.poseModel === name && this.pose) return;
    onStep("Loading the " + name + " body model…");
    const next = await this.make(PoseLandmarker, { baseOptions: { modelAssetPath: MODEL(POSE_MODELS[name] || POSE_MODELS.full) },
      runningMode: "VIDEO", numPoses: 1, minPoseDetectionConfidence: 0.5, minPosePresenceConfidence: 0.5,
      minTrackingConfidence: 0.5, outputSegmentationMasks: true });
    const old = this.pose; this.pose = next; this.poseModel = name;
    if (old) try { old.close(); } catch (e) {}
  }

  // Timestamps given to MediaPipe must always increase, or it throws (a cause of the old app's freezes).
  stamp() { const t = Math.max(this.lastTs + 1, Math.floor(performance.now())); this.lastTs = t; return t; }

  /** Runs the trackers on the current video frame. Returns null if nothing could be read. */
  detect(video) {
    if (!this.pose) return null;
    const t0 = performance.now(), ts = this.stamp();
    const out = { ts, pose: null, world: null, mask: null, maskW: 0, maskH: 0, hands: [] };
    const r = this.pose.detectForVideo(video, ts);
    if (r.landmarks && r.landmarks.length) { out.pose = r.landmarks[0]; out.world = r.worldLandmarks && r.worldLandmarks[0]; }
    if (r.segmentationMasks && r.segmentationMasks.length) {
      const m = r.segmentationMasks[0];
      if (this.wantMask && out.pose) { out.mask = m.getAsFloat32Array(); out.maskW = m.width; out.maskH = m.height; }
      for (const mm of r.segmentationMasks) try { mm.close(); } catch (e) {}
    }
    if (this.hand && this.wantHands && out.pose) {
      const h = this.hand.detectForVideo(video, this.stamp());
      const lms = h.landmarks || [];
      for (let i = 0; i < lms.length; i++) out.hands.push({ lm: lms[i], score: (h.handedness && h.handedness[i] && h.handedness[i][0]) ? h.handedness[i][0].score : 1 });
    }
    this.inferMs = this.inferMs * 0.85 + (performance.now() - t0) * 0.15;
    this.errors = 0;
    return out;
  }

  close() { for (const t of [this.pose, this.hand]) if (t) try { t.close(); } catch (e) {} this.pose = this.hand = null; }
}

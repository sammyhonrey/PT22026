"""
See your Life X-ray - anatomy data builder
==========================================

Turns the BodyParts3D meshes (one adult male, STL files named by FMA id) into the
compact, rigged data files that the app loads from web/assets/anatomy/.

What it does, step by step:
  1. picks the structures for each layer (skeleton, muscles, heart, vessels, lungs,
     organs, brain, skin) from the BodyParts3D part list,
  2. converts millimetres in BodyParts3D axes to metres in the app's axes
     (x = body's left, y = up, z = forward),
  3. measures the rest position of every joint the trackers report (33 body points
     and 21 points per hand) from the bones themselves,
  4. simplifies every mesh to a polygon budget so it runs in real time,
  5. "skins" every vertex to one or two of 45 body segments (pelvis, chest, head,
     arms, hands, fingers, legs, feet), so the anatomy bends with the person,
  6. writes one binary file per layer plus manifest.json (layers, joints, labels).

Run (only needed to rebuild the data; the built files are in web/assets/anatomy):
    pip install -r requirements-dev.txt
    python scripts/build_anatomy.py --stl PATH/TO/BodyParts3D/stl --parts PATH/TO/parts_list_e.txt --out web/assets/anatomy

BodyParts3D, (c) The Database Center for Life Science, licensed under
CC Attribution-Share Alike 2.1 Japan. The output meshes are adapted (simplified,
re-oriented, rigged) from it and are shared under the same licence.
"""
import argparse, json, os, re, struct, sys
import numpy as np
import trimesh
import fast_simplification

# ----------------------------------------------------------------------------- parts
PARTS_TSV_HELP = "parts list: FMAID<TAB>english name (BodyParts3D parts_list_e.txt)"

EXCLUDE = re.compile(r"penis|testis|epididym|deferent|seminal|prostate|pubic hair|glans|urethra|head hairs|"
                     r"eyebrows|pubococc|puborect|iliococc|coccygeus|levator ani|anal sphincter|^ear$|labial part|gingiva")

def layer_of(n):
    n = n.lower()
    if EXCLUDE.search(n): return None
    if n == "skin": return "skin"
    if re.search(r"heart|coronary|cardiac|valve|papillary|coronary sinus|veins of left ventricle", n): return "heart"
    if re.search(r"aorta|vena cava|artery|arteries|vein\b|veins|carotid|jugular|subclavian|brachiocephalic", n): return "vessels"
    if re.search(r"lung|trachea|bronch|thymus|diaphragm", n): return "lungs"
    if re.search(r"liver|stomach|spleen|pancrea|gallbladder|duoden|jejun|ileum|colon|rectum|appendix|esophag|taenia|"
                 r"kidney|ureter|bladder|adrenal", n): return "organs"
    if re.search(r"gyrus|gyri|lobe|lobule|cerebell|pons|medulla|midbrain|thalam|hippocamp|amygdala|caudate|putamen|"
                 r"pallidus|capsule|fornix|corpus callosum|commissure|ventricle|aqueduct|optic|geniculate|colliculus|"
                 r"insula|habenula|pineal|pituitary|chiasm|tuber cinereum|mammillary|septum pellucidum|lamina terminalis|"
                 r"choroid|white matter|stria|peduncle|interpeduncular|interventricular foramen|central canal|precuneus|eyeball", n):
        return "brain"
    if re.search(r"bone|vertebra|rib\b|costal cartilage|femur|tibia|fibula|humerus|radius|ulna|scapula|clavicle|patella|"
                 r"phalanx|metacarpal|metatarsal|sacrum|^atlas|^axis|talus|calcaneus|cuneiform|cuboid|navicular|scaphoid|"
                 r"lunate|triquetral|pisiform|trapezium|trapezoid|capitate|hamate|sesamoid|mandible|maxilla|ethmoid|vomer|"
                 r"concha|manubrium|sternum|xiphoid|disk|tooth|hyoid|thyroid cartilage|nasal cartilages", n) and "membrane" not in n:
        return "skeleton"
    return "muscles"

# Polygon budget per layer (triangles) and the colours used by the "Anatomical" look.
BUDGET = {"skeleton": 150000, "muscles": 170000, "skin": 40000, "heart": 36000, "vessels": 22000,
          "lungs": 30000, "organs": 46000, "brain": 40000}
MIN_TRIS = 80

def tone(name, layer):
    n = name.lower()
    if layer == "skeleton":
        if "tooth" in n: return (245, 244, 236)
        if "disk" in n or "cartilage" in n: return (150, 196, 214)
        return (232, 222, 196)
    if layer == "muscles":
        if re.search(r"aponeurosis|linea alba|retinaculum|ligament|tract|tendon|membrane", n): return (226, 214, 190)
        return (186, 58, 52)
    if layer == "heart":
        if "valve" in n: return (240, 214, 120)
        if "vein" in n or "sinus" in n: return (70, 110, 220)
        if "artery" in n or "arteries" in n or "branch" in n: return (240, 70, 60)
        if "papillary" in n: return (205, 85, 95)
        return (196, 40, 52)
    if layer == "vessels":
        if "vein" in n or "vena" in n: return (64, 104, 214)
        return (226, 48, 44)
    if layer == "lungs":
        if "diaphragm" in n: return (176, 72, 64)
        if "trachea" in n or "bronch" in n: return (214, 196, 170)
        if "thymus" in n: return (220, 170, 150)
        return (236, 150, 160)
    if layer == "organs":
        if "liver" in n: return (140, 46, 40)
        if "gallbladder" in n: return (90, 150, 70)
        if "spleen" in n: return (120, 40, 70)
        if "kidney" in n: return (150, 56, 50)
        if "adrenal" in n: return (220, 170, 80)
        if "stomach" in n or "esophag" in n: return (222, 130, 120)
        if "pancrea" in n: return (230, 190, 130)
        if "bladder" in n or "ureter" in n: return (230, 200, 120)
        return (226, 160, 140)
    if layer == "brain":
        if "eyeball" in n: return (244, 244, 238)
        if "cerebell" in n: return (220, 150, 140)
        if re.search(r"pons|medulla|midbrain|peduncle", n): return (226, 186, 150)
        if "ventricle" in n or "aqueduct" in n or "canal" in n: return (110, 170, 230)
        return (232, 168, 172)
    return (230, 200, 180)

# ----------------------------------------------------------------------------- helpers
def to_app(p):
    """BodyParts3D mm (x = body's left, y = back, z = up) -> metres (x = left, y = up, z = forward)."""
    p = np.asarray(p, dtype=np.float64)
    return np.stack([p[..., 0], p[..., 2], -p[..., 1]], axis=-1) / 1000.0

def load(stl_dir, fid):
    m = trimesh.load(os.path.join(stl_dir, fid + ".stl"), process=True)
    return to_app(m.vertices).astype(np.float32), m.faces.astype(np.int32)

def ends(v):
    """Both ends of a long bone along its main axis: (end A, end B) as centroids of the outer 8%."""
    c = v.mean(0); u, s, vt = np.linalg.svd(v - c, full_matrices=False); ax = vt[0]
    t = (v - c) @ ax; lo, hi = t.min(), t.max(); span = hi - lo
    return v[t < lo + span * 0.08].mean(0), v[t > hi - span * 0.08].mean(0)

def seg_dist(p, a, b):
    ab = b - a; t = np.clip(((p - a) @ ab) / max(ab @ ab, 1e-12), 0, 1)
    return np.linalg.norm(p - (a + t[:, None] * ab), axis=1), t

def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)

# ----------------------------------------------------------------------------- segments
# 0 pelvis, 1 chest, 2 head; arms L 3,4,5 R 6,7,8; legs L 9,10,11 R 12,13,14;
# finger bones L 15..29, R 30..44 (thumb: metacarpal, proximal, distal; others: proximal, middle, distal)
SEG = {"pelvis": 0, "chest": 1, "head": 2}
for side, base in (("L", 3), ("R", 6)):
    SEG[side + "uarm"], SEG[side + "farm"], SEG[side + "palm"] = base, base + 1, base + 2
for side, base in (("L", 9), ("R", 12)):
    SEG[side + "thigh"], SEG[side + "shin"], SEG[side + "foot"] = base, base + 1, base + 2
FINGERS = ["thumb", "index", "middle", "ring", "little"]
for side, base in (("L", 15), ("R", 30)):
    for fi, f in enumerate(FINGERS):
        for k in range(3): SEG[f"{side}{f}{k}"] = base + fi * 3 + k
NSEG = 45
# hand landmark chains (MediaPipe hand: 0 wrist, thumb 1-4, index 5-8, middle 9-12, ring 13-16, little 17-20)
FINGER_LM = {"thumb": [1, 2, 3, 4], "index": [5, 6, 7, 8], "middle": [9, 10, 11, 12], "ring": [13, 14, 15, 16], "little": [17, 18, 19, 20]}

def side_of(n):
    n = n.lower()
    if re.search(r"\bleft\b", n): return "L"
    if re.search(r"\bright\b", n): return "R"
    return None

ORD = {"first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5}
FING_NAME = {"thumb": "thumb", "index finger": "index", "middle finger": "middle", "ring finger": "ring", "little finger": "little"}

def rigid_segment(name, layer):
    """Bones that move as one piece: returns a segment name or None (= skin per vertex)."""
    if layer != "skeleton": return None
    n = name.lower(); s = side_of(n)
    if s:
        if "humerus" in n: return s + "uarm"
        if "radius" in n or "ulna" in n: return s + "farm"
        if re.search(r"scaphoid|lunate|triquetral|pisiform|trapezium|trapezoid|capitate|hamate", n): return s + "palm"
        m = re.search(r"(first|second|third|fourth|fifth) metacarpal", n)
        if m: return s + ("thumb0" if m.group(1) == "first" else "palm")
        m = re.search(r"(proximal|middle|distal) phalanx of (left|right) (thumb|index finger|middle finger|ring finger|little finger)", n)
        if m:
            f = FING_NAME[m.group(3)]; k = {"proximal": 0, "middle": 1, "distal": 2}[m.group(1)]
            if f == "thumb": k = {"proximal": 1, "distal": 2}[m.group(1)]
            return f"{s}{f}{k}"
        if "femur" in n or "patella" in n: return s + "thigh"
        if "tibia" in n or "fibula" in n: return s + "shin"
        if re.search(r"talus|calcaneus|navicular|cuneiform|cuboid|metatarsal|toe|sesamoid bone of", n): return s + "foot"
        if "hip bone" in n: return "pelvis"
        if "clavicle" in n or "scapula" in n: return "chest"
    if "sacrum" in n: return "pelvis"
    if re.search(r"frontal bone|occipital bone|sphenoid|temporal bone|ethmoid|mandible|parietal bone|zygomatic|lacrimal|"
                 r"nasal bone|maxilla|palatine|concha|vomer|tooth|nasal cartilages|^atlas|^axis|intervertebral disk of axis", n):
        return "head"
    return None   # vertebrae, ribs, sternum, discs, hyoid...: smooth trunk blend per vertex

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stl", required=True, help="folder with BodyParts3D FMA*.stl files")
    ap.add_argument("--parts", required=True, help=PARTS_TSV_HELP)
    ap.add_argument("--out", required=True)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply every polygon budget")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    names = {}
    for line in open(a.parts, encoding="utf-8"):
        f = line.rstrip("\n").split("\t")
        if len(f) >= 2 and f[0].startswith("FMA"): names[f[0]] = f[1]
    avail = {f[:-4] for f in os.listdir(a.stl) if f.endswith(".stl")}
    by_name = {names[f]: f for f in avail if f in names}
    parts = {}
    for fid in sorted(avail):
        if fid not in names: continue
        L = layer_of(names[fid])
        if L: parts.setdefault(L, []).append((fid, names[fid]))
    print({k: len(v) for k, v in parts.items()})

    cache = {}
    def mesh(name):
        if name not in cache: cache[name] = load(a.stl, by_name[name])
        return cache[name]
    def V(name): return mesh(name)[0]

    # ---------------- rest joints (metres, app axes) ----------------
    J = {}
    for s, w in (("L", "left"), ("R", "right")):
        hum = V(f"{w} humerus"); top = hum[:, 1].max(); bot = hum[:, 1].min()
        J[s + "shoulder"] = hum[hum[:, 1] > top - 0.045].mean(0)
        J[s + "elbow"] = hum[hum[:, 1] < bot + 0.022].mean(0)
        ra, ul = V(f"{w} radius"), V(f"{w} ulna"); fa = np.vstack([ra, ul]); fb = fa[:, 1].min()
        J[s + "wrist"] = fa[fa[:, 1] < fb + 0.018].mean(0)
        fem = V(f"{w} femur"); ftop = fem[:, 1].max(); fbot = fem[:, 1].min()
        topz = fem[fem[:, 1] > ftop - 0.06]
        med = topz[np.abs(topz[:, 0]) < np.median(np.abs(topz[:, 0]))]           # femoral head (medial)
        lat = topz[np.abs(topz[:, 0]) >= np.percentile(np.abs(topz[:, 0]), 85)]  # greater trochanter (lateral)
        J[s + "hip"] = 0.6 * med.mean(0) + 0.4 * lat.mean(0)
        tib = V(f"{w} tibia")
        J[s + "knee"] = 0.5 * fem[fem[:, 1] < fbot + 0.025].mean(0) + 0.5 * tib[tib[:, 1] > tib[:, 1].max() - 0.02].mean(0)
        fib = V(f"{w} fibula")
        J[s + "ankle"] = 0.5 * tib[tib[:, 1] < tib[:, 1].min() + 0.012].mean(0) + 0.5 * fib[fib[:, 1] < fib[:, 1].min() + 0.012].mean(0)
        cal = V(f"{w} calcaneus")
        back = cal[cal[:, 2] < cal[:, 2].min() + 0.012]; J[s + "heel"] = back[back[:, 1] < np.median(back[:, 1])].mean(0)
        toes = np.vstack([V(n) for n in by_name if re.search(rf"phalanx of {w} (big|second) toe", n)])
        J[s + "foot_index"] = toes[toes[:, 2] > toes[:, 2].max() - 0.01].mean(0)
        # ---- hand: 21 points from the bones' ends (proximal end = the one nearer the wrist)
        wr = J[s + "wrist"]; H = [None] * 21; H[0] = wr
        def pd(name):
            e1, e2 = ends(V(name)); return (e1, e2) if np.linalg.norm(e1 - wr) < np.linalg.norm(e2 - wr) else (e2, e1)
        mc = {k: pd(f"{w} {['first', 'second', 'third', 'fourth', 'fifth'][k - 1]} metacarpal bone") for k in range(1, 6)}
        H[1], H[2] = mc[1]
        p1, d1 = pd(f"proximal phalanx of {w} thumb")[1], pd(f"distal phalanx of {w} thumb")
        H[3] = 0.5 * (p1 + d1[0]); H[4] = d1[1]
        for fi, f in enumerate(["index finger", "middle finger", "ring finger", "little finger"]):
            b = 5 + fi * 4; pr, mi, di = pd(f"proximal phalanx of {w} {f}"), pd(f"middle phalanx of {w} {f}"), pd(f"distal phalanx of {w} {f}")
            H[b] = 0.5 * (mc[fi + 2][1] + pr[0]); H[b + 1] = 0.5 * (pr[1] + mi[0]); H[b + 2] = 0.5 * (mi[1] + di[0]); H[b + 3] = di[1]
        J[s + "hand"] = np.array(H)
        J[s + "index"] = 0.5 * (H[5] + H[6]); J[s + "pinky"] = 0.5 * (H[17] + H[18]); J[s + "thumb"] = 0.5 * (H[2] + H[3])

    eye = V("eyeball"); ear = V("ear"); skin = V("skin")
    for s, sg in (("L", 1), ("R", -1)):
        e = eye[eye[:, 0] * sg > 0]; c = e.mean(0); front = e[e[:, 2] > e[:, 2].max() - 0.004].mean(0)
        J[s + "eye"] = front; w = e[:, 0].max() - e[:, 0].min()
        J[s + "eye_inner"] = front - np.array([sg * w * 0.45, 0, 0.002]); J[s + "eye_outer"] = front + np.array([sg * w * 0.45, 0, -0.004])
        J[s + "ear"] = ear[ear[:, 0] * sg > 0].mean(0)
        oo = V("orbicularis oris"); corner = oo[np.argmax(oo[:, 0] * sg)]; J[s + "mouth"] = corner
    face = skin[(np.abs(skin[:, 0]) < 0.006) & (skin[:, 1] > J["Leye"][1] - 0.07) & (skin[:, 1] < J["Leye"][1] - 0.01)]
    J["nose"] = face[np.argmax(face[:, 2])]

    order = ["nose", "Leye_inner", "Leye", "Leye_outer", "Reye_inner", "Reye", "Reye_outer", "Lear", "Rear", "Lmouth", "Rmouth",
             "Lshoulder", "Rshoulder", "Lelbow", "Relbow", "Lwrist", "Rwrist", "Lpinky", "Rpinky", "Lindex", "Rindex", "Lthumb", "Rthumb",
             "Lhip", "Rhip", "Lknee", "Rknee", "Lankle", "Rankle", "Lheel", "Rheel", "Lfoot_index", "Rfoot_index"]
    POSE = np.array([J[k] for k in order])
    print("rest shoulders", POSE[11], POSE[12], "hips", POSE[23], POSE[24])

    # ---------------- skinning geometry ----------------
    # limb "bones" (lines) per side, with a radius (m) used for distance
    LIMBS = []
    for s in "LR":
        Hh = J[s + "hand"]; knuck = (Hh[5] + Hh[17]) * 0.5
        LIMBS += [(SEG[s + "uarm"], J[s + "shoulder"], J[s + "elbow"], 0.040, s, "arm"),
                  (SEG[s + "farm"], J[s + "elbow"], J[s + "wrist"], 0.032, s, "arm"),
                  (SEG[s + "palm"], J[s + "wrist"], knuck, 0.030, s, "arm"),
                  (SEG[s + "thigh"], J[s + "hip"], J[s + "knee"], 0.070, s, "leg"),
                  (SEG[s + "shin"], J[s + "knee"], J[s + "ankle"], 0.045, s, "leg"),
                  (SEG[s + "foot"], J[s + "heel"] + np.array([0, 0.03, 0]), J[s + "foot_index"], 0.035, s, "leg")]
        for f, ch in FINGER_LM.items():
            for k in range(3):
                LIMBS.append((SEG[f"{s}{f}{k}"], Hh[ch[k]], Hh[ch[k + 1]], 0.009, s, "finger"))

    # trunk outline per height from trunk bones (half width x, centre & half depth z)
    trunk_bones = [n for n in by_name if layer_of(n) == "skeleton" and re.search(r"rib\b|vertebra|sternum|manubrium|xiphoid|hip bone|sacrum|costal", n)]
    TB = np.vstack([V(n) for n in trunk_bones])
    ys = np.arange(0.70, 1.62, 0.01); prof = []
    for y in ys:
        sl = TB[np.abs(TB[:, 1] - y) < 0.02]
        if len(sl) < 20: prof.append(None); continue
        prof.append((np.percentile(np.abs(sl[:, 0]), 98) + 0.015, 0.5 * (sl[:, 2].max() + sl[:, 2].min()), 0.5 * (sl[:, 2].max() - sl[:, 2].min()) + 0.03))
    good = [i for i, p in enumerate(prof) if p is not None]
    prof_arr = np.array([prof[i] if prof[i] is not None else prof[min(good, key=lambda g: abs(g - i))] for i in range(len(ys))])
    neck_w = 0.06
    def trunk_dist(p):
        idx = np.clip(np.round((p[:, 1] - 0.70) / 0.01).astype(int), 0, len(ys) - 1)
        hw, zc, hd = prof_arr[idx, 0], prof_arr[idx, 1], prof_arr[idx, 2]
        hw = np.where(p[:, 1] > J["Lshoulder"][1] + 0.02, neck_w, hw)
        r = np.sqrt((p[:, 0] / hw) ** 2 + ((p[:, 2] - zc) / hd) ** 2)
        return np.maximum(0, r - 1) * hw

    y_hip = 0.5 * (J["Lhip"][1] + J["Rhip"][1]); y_sh = 0.5 * (J["Lshoulder"][1] + J["Rshoulder"][1])
    y_c7 = y_sh + 0.045; y_c2 = 0.5 * (J["Lear"][1] + J["Rear"][1]) - 0.035
    def trunk_split(p):
        """weights over (pelvis, chest, head) by height"""
        wc = smoothstep(y_hip + 0.16, y_hip + 0.36, p[:, 1]); wh = smoothstep(y_c7, y_c2, p[:, 1])
        return np.stack([1 - wc, wc * (1 - wh), wc * wh], 1)

    def skin_weights(p, side_hint, layer):
        """Per vertex: up to 2 segments + weight of the first."""
        n = len(p); tau = 0.012
        cands_d = [trunk_dist(p)]; cands_s = [None]
        for sid, a0, b0, rad, s, kind in LIMBS:
            d, t = seg_dist(p, a0, b0); d = np.maximum(0, d - rad)
            side_ok = (p[:, 0] > 0) if s == "L" else (p[:, 0] < 0)
            if side_hint is not None: side_ok = np.full(n, side_hint == s)
            d = np.where(side_ok, d, 9.0)
            if kind == "finger": d = d + 0.002   # prefer palm when equal
            cands_d.append(d); cands_s.append(sid)
        D = np.stack(cands_d, 1)
        order_ = np.argsort(D, 1)[:, :2]
        d1 = np.take_along_axis(D, order_[:, :1], 1)[:, 0]; d2 = np.take_along_axis(D, order_[:, 1:2], 1)[:, 0]
        w1 = 1 / (1 + np.exp(-(d2 - d1) / tau))             # soft choice between the two nearest
        # expand: the trunk candidate is itself split over pelvis/chest/head
        TS = trunk_split(p)
        W = np.zeros((n, NSEG), np.float32)
        for slot, w in ((order_[:, 0], w1), (order_[:, 1], 1 - w1)):
            is_tr = slot == 0
            W[is_tr, 0] += w[is_tr] * TS[is_tr, 0]; W[is_tr, 1] += w[is_tr] * TS[is_tr, 1]; W[is_tr, 2] += w[is_tr] * TS[is_tr, 2]
            for ci in np.unique(slot[~is_tr]):
                m = slot == ci; W[m, cands_s[ci]] += w[m]
        return W

    def pack2(W):
        o = np.argsort(-W, 1)[:, :2]; w = np.take_along_axis(W, o, 1)
        w0 = w[:, 0] / np.maximum(w.sum(1), 1e-9)
        return o.astype(np.uint8), np.round(w0 * 255).astype(np.uint8)

    # ---------------- per layer: decimate, skin, write ----------------
    manifest = {"version": 1, "units": "metres; x = body's left, y = up, z = forward",
                "credit": "BodyParts3D, (c) The Database Center for Life Science, CC Attribution-Share Alike 2.1 Japan. "
                          "Meshes simplified, re-oriented and rigged for See your Life X-ray.",
                "segments": NSEG, "pose_rest": np.round(POSE, 5).tolist(),
                "hand_rest": {s: np.round(J[s + "hand"], 5).tolist() for s in "LR"}, "layers": [], "labels": []}
    LABELS = json.load(open(os.path.join(os.path.dirname(__file__), "labels.json"), encoding="utf-8"))
    for layer in ["skin", "skeleton", "muscles", "heart", "vessels", "lungs", "organs", "brain"]:
        plist = parts.get(layer, [])
        if not plist: continue
        meshes = []
        areas = []
        for fid, name in plist:
            v, f = load(a.stl, fid); meshes.append((fid, name, v, f))
            areas.append(trimesh.triangles.area(v[f]).sum())
        areas = np.array(areas); budget = BUDGET[layer] * a.scale
        Vs, Fs, Ss, Ws, Cs, Ps = [], [], [], [], [], []; off = 0; part_info = []
        for (fid, name, v, f), ar in zip(meshes, areas):
            want = max(MIN_TRIS, int(budget * ar / areas.sum()))
            if len(f) > want * 1.05:
                v2, f2 = fast_simplification.simplify(v, f, target_reduction=1 - want / len(f))
            else: v2, f2 = v, f
            if len(f2) == 0: continue
            v2 = v2.astype(np.float32)
            rs = rigid_segment(name, layer)
            if rs is not None:
                W = np.zeros((len(v2), NSEG), np.float32); W[:, SEG[rs]] = 1
            elif layer == "skeleton" or layer in ("heart", "lungs", "organs", "brain"):
                W = np.zeros((len(v2), NSEG), np.float32); TS = trunk_split(v2); W[:, 0:3] = TS
            else:
                W = skin_weights(v2, side_of(name) if layer != "skin" else None, layer)
            si, sw = pack2(W)
            pid = len(part_info)
            part_info.append({"id": fid, "name": name, "v0": off, "nv": len(v2)})
            Vs.append(v2); Fs.append(f2 + off); Ss.append(si); Ws.append(sw)
            Cs.append(np.tile(np.array(tone(name, layer), np.uint8), (len(v2), 1))); Ps.append(np.full(len(v2), pid, np.uint16))
            off += len(v2)
        V_ = np.vstack(Vs); F_ = np.vstack(Fs).astype(np.uint32); S_ = np.vstack(Ss); W_ = np.concatenate(Ws); C_ = np.vstack(Cs); P_ = np.concatenate(Ps)
        lo = V_.min(0); size = V_.max(0) - lo
        q = np.round((V_ - lo) / size * 65535).astype(np.uint16)
        fn = f"{layer}.mesh"
        with open(os.path.join(a.out, fn), "wb") as fh:
            blobs = [q.tobytes(), S_.tobytes(), W_.tobytes(), C_.tobytes(), P_.tobytes(), F_.tobytes()]
            offs = []; pos = 0
            for b in blobs:
                offs.append(pos); fh.write(b); pos += len(b)
                pad = (-pos) % 4; fh.write(b"\0" * pad); pos += pad
        manifest["layers"].append({"name": layer, "file": fn, "nv": int(len(V_)), "nt": int(len(F_)),
                                   "min": np.round(lo, 6).tolist(), "size": np.round(size, 6).tolist(),
                                   "offsets": {"pos": offs[0], "seg": offs[1], "w": offs[2], "col": offs[3], "part": offs[4], "idx": offs[5]},
                                   "parts": [{"n": p["name"], "v0": p["v0"], "nv": p["nv"]} for p in part_info]})
        print(f"{layer:9s} parts={len(part_info):4d} verts={len(V_):7d} tris={len(F_):7d} bytes={os.path.getsize(os.path.join(a.out, fn))}")
        # labels anchored on a vertex of the named parts
        for lab in LABELS:
            if lab["layer"] != layer: continue
            for pn in lab["parts"]:
                pi = next((i for i, p in enumerate(part_info) if p["name"] == pn), None)
                if pi is None: print("  label part missing:", pn); continue
                p = part_info[pi]; vv = V_[p["v0"]:p["v0"] + p["nv"]]; c = vv.mean(0)
                k = p["v0"] + int(np.argmin(np.linalg.norm(vv - c, axis=1) - 0.6 * (vv[:, 2] - c[2])))  # central, front-facing vertex
                manifest["labels"].append({"t": lab["t"], "layer": layer, "p": np.round(V_[k], 5).tolist(),
                                           "s": [int(S_[k, 0]), int(S_[k, 1])], "w": round(float(W_[k]) / 255, 3),
                                           "side": side_of(pn) or "", "info": lab.get("info", "")})
    with open(os.path.join(a.out, "manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, separators=(",", ":"))
    print("wrote", os.path.join(a.out, "manifest.json"))

if __name__ == "__main__":
    main()

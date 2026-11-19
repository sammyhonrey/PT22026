// See your Life X-ray - live 3D author badge (top bar of the running app)
import * as THREE from "three";
import { FontLoader } from "three/addons/loaders/FontLoader.js";
import { TextGeometry } from "three/addons/geometries/TextGeometry.js";

export const AUTHOR = "Mohammad Kazim Honarjo";

export class Badge {
  constructor(canvas) {
    this.canvas = canvas;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setClearColor(0x000000, 0);
    this.scene = new THREE.Scene();
    this.cam = new THREE.PerspectiveCamera(30, 3.2, 0.1, 100); this.cam.position.set(0, 0, 6.3);
    this.group = new THREE.Group(); this.scene.add(this.group);
    this.scene.add(new THREE.AmbientLight(0x8fb4ff, 0.55));
    const k = new THREE.DirectionalLight(0xffffff, 2.2); k.position.set(2, 3, 5); this.scene.add(k);
    this.rimA = new THREE.PointLight(0x2dd4bf, 30, 20); this.rimA.position.set(-4, 1, 2); this.scene.add(this.rimA);
    this.rimB = new THREE.PointLight(0xf472b6, 24, 20); this.rimB.position.set(4, -1, 2); this.scene.add(this.rimB);
    this.ready = false; this.textWidth = 9; this.t0 = performance.now();
  }

  async load(fontUrl) {
    const font = await new FontLoader().loadAsync(fontUrl);
    const mk = (text, size, depth, bevel) => {
      const g = new TextGeometry(text, { font, size, depth, curveSegments: 6, bevelEnabled: bevel > 0, bevelThickness: bevel, bevelSize: bevel * 0.6, bevelSegments: 3 });
      g.computeBoundingBox(); const b = g.boundingBox; g.translate(-(b.max.x + b.min.x) / 2, -(b.max.y + b.min.y) / 2, -depth / 2); return g;
    };
    const metal = new THREE.MeshStandardMaterial({ color: 0xffffff, metalness: 0.25, roughness: 0.3, emissive: 0xbfeaff, emissiveIntensity: 0.55 });
    const side = new THREE.MeshStandardMaterial({ color: 0x14b8a6, metalness: 0.6, roughness: 0.35, emissive: 0x0b5d57, emissiveIntensity: 0.9 });
    this.name = new THREE.Mesh(mk(AUTHOR, 0.42, 0.16, 0.025), [metal, side]);
    this.name.position.y = 0.22; this.group.add(this.name);
    const sub = new THREE.MeshStandardMaterial({ color: 0x9fe8ff, metalness: 0.3, roughness: 0.5, emissive: 0x1d6f8a, emissiveIntensity: 0.8 });
    this.sub = new THREE.Mesh(mk("AUTHOR  ·  SEE YOUR LIFE X-RAY", 0.15, 0.04, 0), sub);
    this.sub.position.y = -0.38; this.group.add(this.sub);
    // a thin glowing ring that orbits the name
    this.ring = new THREE.Mesh(new THREE.TorusGeometry(3.3, 0.012, 8, 160), new THREE.MeshBasicMaterial({ color: 0x5eead4, transparent: true, opacity: 0.55 }));
    this.ring.scale.set(1, 0.24, 1); this.group.add(this.ring);
    const box = this.name.geometry.boundingBox; this.textWidth = box.max.x - box.min.x; this.fit();
    this.ready = true;
  }

  resize(w, h, dpr) { this.renderer.setPixelRatio(dpr); this.renderer.setSize(w, h, false); this.cam.aspect = w / h; this.fit(); }

  fit() {
    const z = this.cam.position.z, wide = this.textWidth * 1.16 / (2 * this.cam.aspect * z), tall = 0.72 / z;
    this.cam.fov = Math.min(30, THREE.MathUtils.radToDeg(2 * Math.atan(Math.max(wide, tall))));
    this.cam.updateProjectionMatrix();
  }

  render() {
    if (!this.ready) return;
    const t = (performance.now() - this.t0) / 1000;
    this.group.rotation.y = Math.sin(t * 0.6) * 0.26;
    this.group.rotation.x = Math.sin(t * 0.43) * 0.08;
    this.group.position.y = Math.sin(t * 1.1) * 0.06;
    this.ring.rotation.z = t * 0.5; this.ring.rotation.x = 1.25 + Math.sin(t * 0.7) * 0.1;
    this.rimA.position.x = Math.sin(t * 0.9) * 5; this.rimB.position.x = -Math.sin(t * 0.9) * 5;
    this.renderer.render(this.scene, this.cam);
  }
}

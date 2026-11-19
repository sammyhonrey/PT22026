"""Download the files that are too large for the repository.

  - MediaPipe pose and hand landmarker models (Google, Apache 2.0)
  - three.js r170 (MIT) and the MediaPipe Tasks Vision web runtime (Apache 2.0)

Files that already exist are kept.
"""

import argparse
import io
import sys
import tarfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from xray import config

VENDOR = config.WEB_DIR / "vendor"
NPM = "https://registry.npmjs.org/{package}/-/{name}-{version}.tgz"
PACKAGES = [
    {
        "package": "three", "name": "three", "version": "0.170.0", "target": VENDOR / "three",
        "files": {
            "package/build/three.module.min.js": "three.module.min.js",
            "package/examples/jsm/loaders/FontLoader.js": "addons/loaders/FontLoader.js",
            "package/examples/jsm/geometries/TextGeometry.js": "addons/geometries/TextGeometry.js",
        },
    },
    {
        "package": "@mediapipe/tasks-vision", "name": "tasks-vision", "version": "0.10.21",
        "target": VENDOR / "mediapipe",
        "files": {
            "package/vision_bundle.mjs": "vision_bundle.mjs",
            "package/wasm/vision_wasm_internal.js": "wasm/vision_wasm_internal.js",
            "package/wasm/vision_wasm_internal.wasm": "wasm/vision_wasm_internal.wasm",
            "package/wasm/vision_wasm_nosimd_internal.js": "wasm/vision_wasm_nosimd_internal.js",
            "package/wasm/vision_wasm_nosimd_internal.wasm": "wasm/vision_wasm_nosimd_internal.wasm",
        },
    },
]


def read(url):
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.read()


def fetch_models(force):
    config.TASK_DIR.mkdir(parents=True, exist_ok=True)
    for name, kind in config.TASK_FILES.items():
        target = config.TASK_DIR / f"{name}.task"
        if target.exists() and not force:
            continue
        print(f"  {target.name}")
        target.write_bytes(read(config.TASK_URL.format(kind=kind, name=name)))


def fetch_package(spec, force):
    wanted = {src: spec["target"] / dst for src, dst in spec["files"].items()}
    if all(path.exists() for path in wanted.values()) and not force:
        return
    print(f"  {spec['package']} {spec['version']}")
    archive = read(NPM.format(**spec))
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
        for src, path in wanted.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(tar.extractfile(src).read())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--force", action="store_true", help="download again even if the files exist")
    args = parser.parse_args()
    try:
        fetch_models(args.force)
        for spec in PACKAGES:
            fetch_package(spec, args.force)
    except (urllib.error.URLError, OSError, KeyError) as exc:
        sys.exit(f"Download failed: {exc}")
    print("Assets are ready.")


if __name__ == "__main__":
    main()

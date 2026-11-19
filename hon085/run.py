"""Start See your Life X-ray and open it in the browser.

    python run.py                 local use
    python run.py --host 0.0.0.0  reachable from other devices or inside a container
"""

import argparse
import sys
import threading
import webbrowser

from xray import config

MIN_PYTHON = (3, 10)


def main():
    if sys.version_info < MIN_PYTHON:
        sys.exit("Python 3.10 or newer is required.")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default=config.HOST)
    parser.add_argument("--port", type=int, default=config.PORT)
    parser.add_argument("--no-browser", action="store_true")
    args = parser.parse_args()

    try:
        import uvicorn
    except ImportError:
        sys.exit("Packages are missing. Run: pip install -r requirements.txt")
    if not (config.WEB_DIR / "vendor" / "three").is_dir() or not any(config.TASK_DIR.glob("*.task")):
        sys.exit("App assets are missing. Run: python scripts/fetch_assets.py")
    if not config.CLASSIFIER_PATH.exists():
        print("Note: the pose classifier is not trained yet. Run: python scripts/train.py")

    url = f"http://{'localhost' if args.host in ('127.0.0.1', '0.0.0.0') else args.host}:{args.port}"
    print(f"See your Life X-ray is running at {url}  (press Ctrl+C to stop)")
    if not args.no_browser:
        threading.Timer(1.2, webbrowser.open, args=(url,)).start()
    uvicorn.run("xray.server:app", host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

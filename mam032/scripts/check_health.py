#!/usr/bin/env python3
"""Check one explicit health endpoint without extra runtime dependencies."""

import argparse
import urllib.error
import urllib.request


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", nargs="?", default="http://127.0.0.1:8501/_stcore/health")
    args = parser.parse_args()
    try:
        with urllib.request.urlopen(args.url, timeout=3) as response:
            if response.status != 200 or response.read(32).strip() != b"ok":
                raise ValueError("Health endpoint did not return HTTP 200 / ok.")
    except (OSError, ValueError, urllib.error.URLError) as exc:
        raise SystemExit("Health check failed: " + str(exc)) from exc
    print("Healthy")


if __name__ == "__main__":
    main()

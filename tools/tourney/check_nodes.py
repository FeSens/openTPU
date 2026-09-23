"""Checks that every test node id in components/*.yaml exists (pytest --collect-only)."""
import subprocess
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent


def main():
    nodes = set()
    for p in sorted((HERE / "components").glob("*.yaml")):
        c = yaml.safe_load(p.read_text())
        nodes |= set(c["tests"]["fast"]) | set(c["tests"]["board"])
    r = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q",
                        "-p", "no:cacheprovider", *sorted(nodes)],
                       capture_output=True, text=True, cwd=HERE.parents[1])
    print(r.stdout[-800:], r.stderr[-800:])
    sys.exit(r.returncode)


if __name__ == "__main__":
    main()

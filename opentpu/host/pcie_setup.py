"""otpu-setup: the host PC setup for the card (XDMA driver, udev rules, driver options).

A front end for setup_pcie.sh, which ships in this package with its files (pcie/): the
arguments pass through (--check, --rescan, --uninstall, --poll, --irq, --no-dkms, --src DIR,
--help). The script asks for sudo itself; --check runs without it.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

SCRIPT = Path(__file__).with_name("setup_pcie.sh")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    os.execvp("bash", ["bash", str(SCRIPT), *args])


if __name__ == "__main__":
    sys.exit(main())

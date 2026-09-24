"""Staged bring-up of the openTPU card: a wrapper of opentpu.host.selftest (installed as
otpu-selftest; the stages are listed there).

    python3 tools/board_selftest.py [--sim] [--dev /dev/xdma0] [--qwen models/Qwen3-0.6B]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opentpu.host.selftest import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

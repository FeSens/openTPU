"""Chat with Qwen3 running on openTPU: a wrapper of opentpu.host.chat (installed as otpu-chat).

    python3 tools/chat.py [--backend isa|board|board-sim|rtl] [--prompt ...] [--think]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opentpu.host.chat import Chat, main, make_backend, sampler  # noqa: E402,F401

if __name__ == "__main__":
    main()

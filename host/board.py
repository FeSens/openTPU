"""Shim: `host.board` is `opentpu.host.board` (the same module object, so monkeypatching either
affects both)."""
import sys

from opentpu.host import board as _m

sys.modules[__name__] = _m

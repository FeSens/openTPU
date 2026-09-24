"""Shim: `host.checks` is `opentpu.host.checks` (the same module object, so monkeypatching either
affects both)."""
import sys

from opentpu.host import checks as _m

sys.modules[__name__] = _m

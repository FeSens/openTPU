import numpy as np
import pytest

from opentpu import rtlsim


def rel(a, b):
    return float(np.linalg.norm(np.asarray(a, np.float64) - b) / np.linalg.norm(b))


def assert_same_state(ri, rr):
    """RTL and ISA simulator must agree bit for bit on every slice's DRAM and TMEM."""
    for s, (a, b) in enumerate(zip(ri.drams, rr.drams)):
        bad = np.nonzero(a != b)[0]
        assert len(bad) == 0, f"slice {s}: {len(bad)} DRAM bytes differ, first at {bad[:8]}"
    for s, (a, b) in enumerate(zip(ri.tmems, rr.tmems)):
        bad = np.nonzero(a != b)[0]
        assert len(bad) == 0, f"slice {s}: {len(bad)} TMEM words differ, first at {bad[:8]}"


@pytest.fixture(scope="session")
def have_verilator():
    import shutil
    if shutil.which("verilator") is None:
        pytest.skip("verilator not installed")
    return True

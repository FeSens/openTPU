"""The test session's temporary root (tests/conftest.py): one per pytest process, so gates running
side by side on one host never remove each other's directories."""
import os
from pathlib import Path

import pytest


def test_the_session_has_a_temporary_root_of_its_own(request, tmp_path):
    own = getattr(request.config, "_otpu_basetemp", None)
    if own is None:
        pytest.skip("--basetemp given: the caller chose the root")
    assert Path(own).name == f"pid-{os.getpid()}"
    assert Path(own).resolve() in tmp_path.resolve().parents

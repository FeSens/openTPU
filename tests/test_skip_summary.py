"""The gates' skips by reason (conftest.pytest_terminal_summary): a missing checkpoint, no
Verilator and a slow test each get a line with their count, instead of one "N skipped"."""
from types import SimpleNamespace

from conftest import pytest_terminal_summary, skip_reasons


def _skip(why):
    return SimpleNamespace(longrepr=("tests/test_x.py", 1, f"Skipped: {why}"))


def test_skips_by_reason():
    reports = [_skip("no LFM2 checkpoint"), _skip("verilator not installed"),
               _skip("no LFM2 checkpoint"), _skip("slow: needs --runslow")]
    assert skip_reasons(reports) == [(2, "no LFM2 checkpoint"), (1, "slow: needs --runslow"),
                                     (1, "verilator not installed")]
    lines = []
    tr = SimpleNamespace(stats={"skipped": reports}, write_line=lines.append,
                         write_sep=lambda sep, title: lines.append(title))
    pytest_terminal_summary(tr)
    assert lines == ["4 skipped, by reason", "    2  no LFM2 checkpoint",
                     "    1  slow: needs --runslow", "    1  verilator not installed"]
    lines.clear()
    pytest_terminal_summary(SimpleNamespace(stats={}, write_line=lines.append))
    assert lines == []

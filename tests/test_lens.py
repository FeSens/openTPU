"""openTPU Lens: profile files, recording on the RTL and the ISA simulator, the app page and
its local server."""
import gzip
import json
import re
import shutil
import subprocess
import urllib.request

import pytest

from opentpu import lens


@pytest.fixture(scope="module")
def rtl_prof(have_verilator):
    return lens.record("mlp-small")


def test_record_rtl_has_timeline_and_counters(rtl_prof):
    d = rtl_prof
    assert d["kind"] == "rtl" and d["cycles"] > 0 and d["instrs"]
    assert 0 < d["roofline"]["efficiency"] <= 1.0
    b = d["slices"][0]["buckets"]
    # P and Q trace lines are bucketed together
    for k in ("c", "n", "bm", "bd", "mx", "fm", "bs", "ms", "mb", "ff"):
        assert k in b and len(b[k]) == len(b["c"])
    assert sum(b["n"]) >= d["cycles"] - 2
    # every dynamic instruction points into the program and has a consistent interval
    progs = d["programs"]
    for r in d["instrs"]:
        s, pc, disp, start, end = r[0], r[2], r[6], r[9], r[10]
        assert 0 <= pc < len(progs[s])
        assert disp <= start <= end


def test_file_round_trip(tmp_path, rtl_prof):
    f = lens.save([rtl_prof], tmp_path / "a.otpuprof")
    assert f.read_bytes()[:2] == b"\x1f\x8b"                      # gzip
    doc = lens.load(f)
    assert doc["format"] == lens.FORMAT and doc["version"] == lens.VERSION
    assert doc["profiles"][0] == json.loads(json.dumps(rtl_prof))
    # plain JSON is accepted too
    plain = tmp_path / "b.json"
    plain.write_bytes(gzip.decompress(f.read_bytes()))
    assert lens.load(plain)["profiles"][0]["cycles"] == rtl_prof["cycles"]


def test_rejects_other_files_and_newer_versions(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(json.dumps({"format": "something"}))
    with pytest.raises(ValueError):
        lens.load(p)
    p.write_text(json.dumps({"format": lens.FORMAT, "version": lens.VERSION + 1, "profiles": []}))
    with pytest.raises(ValueError):
        lens.load(p)


def test_isa_profile():
    d = lens.record("mlp-small", isa=True)
    assert d["kind"] == "isa" and d["instrs"] and d["cycles"] > 0
    assert d["roofline"]["bound"] > 0
    # analytic timing: back to back
    rs = [r for r in d["instrs"] if r[0] == 0]
    assert all(a[10] == b[9] for a, b in zip(rs, rs[1:]))


def test_board_profile():
    from opentpu.isasim import board_config
    from opentpu import isa as I
    prog = [I.ld(0, 0, 64), I.halt()]
    d = lens.board_data("board run", board_config(), [prog],
                        {"cycles": 200, "b_reads": 2, "b_writes": 0, "a_reads": 0, "a_writes": 0})
    assert d["kind"] == "board" and d["board"]["cycles"] == 200 and d["roofline"]["bound"] == 2


def test_html_embeds_profile(tmp_path, rtl_prof):
    f = lens.save([rtl_prof], tmp_path / "a.otpuprof")
    html = lens.render(lens.load(f))
    assert "/*__DATA__*/" not in html and '"openTPU-profile"' in html
    assert "</script>" in html and html.count("<script>") == 1


def test_app_javascript_parses(tmp_path):
    node = shutil.which("node")
    if not node:
        pytest.skip("node not installed")
    js = re.search(r"<script>(.*)</script>", lens.APP.read_text(), re.S).group(1)
    f = tmp_path / "app.js"
    f.write_text(js)
    r = subprocess.run([node, "--check", str(f)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr


def test_server_serves_app_and_profile(tmp_path, rtl_prof):
    f = lens.save([rtl_prof], tmp_path / "a.otpuprof")
    srv = lens.serve(f, port=0, open_browser=False, block=False)
    try:
        page = urllib.request.urlopen(srv.url, timeout=10).read().decode()
        assert "openTPU" in page and '"/profile"' in page
        raw = urllib.request.urlopen(srv.url + "profile", timeout=10).read()
        assert raw == f.read_bytes()
    finally:
        srv.shutdown()
        srv.server_close()


CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def test_floorplan_explains_every_element(tmp_path, have_verilator):
    """Headless Chrome: the app's #selftest hovers and pins every floorplan element at several
    moments of a tiny Qwen3 token; every one must show an explanation, with no errors."""
    import html as H
    import os
    chrome = CHROME if os.path.exists(CHROME) else (shutil.which("google-chrome")
                                                     or shutil.which("chromium"))
    if not chrome:
        pytest.skip("Chrome not installed")
    d = lens.record("qwen-tiny", board=True, axi=True, pos=10)
    page = tmp_path / "q.html"
    page.write_text(lens.render(lens.load(lens.save([d], tmp_path / "q.otpuprof"))))
    r = subprocess.run([chrome, "--headless=new", "--disable-gpu", "--window-size=1400,1150",
                        "--virtual-time-budget=60000", "--enable-logging=stderr", "--v=0",
                        "--dump-dom", f"file://{page}#selftest"],
                       capture_output=True, text=True, timeout=300)
    m = re.search(r'<pre id="selftest">(.*?)</pre>', r.stdout, re.S)
    assert m, r.stderr[-2000:]
    res = json.loads(H.unescape(m.group(1)))
    kinds = {k.split(":")[0] for k in res["keys"]}
    assert {"seq", "slot", "dma", "mxu", "fifo", "col", "quant", "act", "tmem", "bank", "vpu",
            "clane", "slane", "coll", "axi", "axiA", "axiB", "dram0", "dram1", "ctrl", "flow",
            "state", "gap", "ctl"} <= kinds
    assert not res["fails"] and not res["errors"], res
    assert not [ln for ln in r.stderr.splitlines() if "Uncaught" in ln]


def test_cli_info(tmp_path, rtl_prof, capsys):
    f = lens.save([rtl_prof], tmp_path / "a.otpuprof")
    lens.main(["info", str(f)])
    out = capsys.readouterr().out
    assert "openTPU-profile v1" in out and rtl_prof["name"] in out

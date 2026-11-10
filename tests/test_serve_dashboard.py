"""otpu-serve's monitor (opentpu/serve/metrics.py, telemetry.py; docs/dashboard.md) over HTTP: a
tiny Qwen3 on the ISA simulator behind the server, the card an in-memory BCU1525 (fake.FakeBcu,
otpu-smi's demo card). The dashboard's page, /metrics in Prometheus' text format (vLLM's names,
histograms cumulative), /api/stats' schema, /api/events' server-sent events, the API key (Bearer
or ?key=), the streamed chat's SSE as OpenAI's clients read it (the openai package's own client
when it is installed), what is n/a left out (no card, no governor, no BMC, a card that fails),
the card's telemetry never in a request's way (a sample stuck on the bus), a prompt's progress
and ETA, and otpu-serve --demo."""
import http.client
import json
import re
import threading
import time
from types import SimpleNamespace

import pytest

from opentpu.host.fake import FakeBcu, FakeTransport
from opentpu.llm.qwen3 import Engine
from opentpu.serve import api as API
from opentpu.serve import metrics as M
from opentpu.serve import scheduler as SC
from opentpu.serve import telemetry as TM

from test_serve_api import _Tok, _call
from test_serve_kvio import _model

torch = pytest.importorskip("torch")

CAP = 512
KEY = "s3cret-key"


class _NoHost:
    """HostStats with nothing to read (a host without /proc or psutil)."""
    def sample(self):
        return {"cpu": None, "ncpu": 1, "mem_total": None, "mem_used": None, "load1": None,
                "proc_cpu": None, "proc_rss": None}


def _serve(srv):
    httpd = API.serve(srv, "127.0.0.1", 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd


@pytest.fixture(scope="module")
def stack():
    """The worker, its monitor (the fake BCU sampled every 0.5 s), a server without a key and
    one with KEY on the same worker."""
    W, spec, cfg = _model("qwen3")
    eng = Engine(spec, W, cap=CAP, cfg=cfg, resident=True, prompt_runs=True)
    tok = _Tok()
    w = SC.Worker(eng, tok, quantum=4).start()
    fk = FakeBcu(run=True, kick_ms=300, run_ms=754_000)
    card = TM.CardSampler("/dev/fakebcu0", 0.5, open_transport=lambda dev: fk)
    mon = M.Monitor(w, "tiny-qwen3", card=card, interval=0.1,
                    info={"backend": "isa", "weights": "fp32", "label": "otpu-serve"})
    defaults = dict(temperature=0.0, top_k=20, top_p=1.0, repetition_penalty=1.0)
    srv = API.Server(w, tok, spec, "tiny-qwen3", defaults, monitor=mon)
    keyed = API.Server(w, tok, spec, "tiny-qwen3", defaults, monitor=mon, api_key=KEY)
    h1, h2 = _serve(srv), _serve(keyed)
    mon.start()
    card.start()
    # one request ended, so the histograms and the table have a row
    st, r = _call(h1.server_address[1], "POST", "/v1/chat/completions",
                  {"messages": [{"role": "user", "content": "hi there"}], "max_tokens": 5,
                   "temperature": 0})
    assert st == 200, r
    t = time.monotonic()
    while card.samples < 2 and time.monotonic() - t < 30:
        time.sleep(0.05)
    yield SimpleNamespace(w=w, mon=mon, card=card, fk=fk, port=h1.server_address[1],
                          kport=h2.server_address[1], spec=spec, tok=tok, defaults=defaults)
    for h in (h1, h2):
        h.shutdown()
        h.server_close()
    mon.stop()
    w.stop()


def _get(port, path, headers=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    c.request("GET", path, headers=headers or {})
    r = c.getresponse()
    body = r.read()
    c.close()
    return r.status, r.getheader("Content-Type", ""), body


# ---------------------------------------------------------------------------- the routes
def test_the_page_and_the_monitor_routes(stack):
    """/ (and /dashboard) is the self-contained page (no script or style from elsewhere);
    /metrics, /api/stats and /api/events answer."""
    for path in ("/", "/dashboard", "/dashboard.html"):
        st, ct, body = _get(stack.port, path)
        assert st == 200 and ct.startswith("text/html"), (path, st)
        page = body.decode()
        assert "OpenTPU" in page and "/api/events" in page
    assert not re.search(r"""<(script|link)[^>]+(src|href)=["']?(https?:)?//""", page)
    st, ct, body = _get(stack.port, "/metrics")
    assert st == 200 and ct == "text/plain; version=0.0.4; charset=utf-8"
    st, ct, body = _get(stack.port, "/api/stats")
    assert st == 200 and ct.startswith("application/json")
    assert _get(stack.port, "/api/nothing")[0] == 404


def test_no_monitor_no_routes():
    """A server without a monitor: its monitor routes are 404, the API as before."""
    srv = API.Server(SimpleNamespace(eng=SimpleNamespace(), tok_lock=threading.Lock()), _Tok(),
                     None, "m", {})
    h = _serve(srv)
    try:
        port = h.server_address[1]
        for path in ("/metrics", "/api/stats", "/api/events"):
            assert _get(port, path)[0] == 404
        assert _get(port, "/")[0] == 200            # (the page says so: n/a everywhere)
    finally:
        h.shutdown()
        h.server_close()


# ---------------------------------------------------------------------------- /api/stats
def test_stats_schema(stack):
    st, _, body = _get(stack.port, "/api/stats?history=1")
    s = json.loads(body)
    for k in ("v", "seq", "t", "server", "queue", "inflight", "throughput", "latency",
              "requests", "kv", "prog_cache", "card", "card_note", "host", "monitor", "point", "recent",
              "series", "card_series"):
        assert k in s, k
    assert s["server"]["model"] == "tiny-qwen3" and s["server"]["cap"] == CAP
    assert s["server"]["backend"] == "isa" and s["server"]["state"] in ("idle", "busy")
    assert set(s["queue"]) == {"running", "waiting", "items"}
    assert set(s["throughput"]) >= {"decode_tps", "prefill_tps", "generated", "fed",
                                    "prompt_tokens", "cached_tokens"}
    assert s["throughput"]["generated"] >= 5
    assert s["throughput"]["prefill_runs"] >= 1     # (the engine's runs feeding the prompts)
    assert s["requests"]["finish_length"] >= 1 and s["latency"]["n"] >= 1
    kv = s["kv"]
    assert kv["cap"] == CAP and 0 < kv["positions"] <= CAP and kv["usage"] == round(
        kv["positions"] / CAP, 5)
    assert set(kv["store"]) == {"ram_used", "ram_limit", "disk_used", "disk_limit",
                                "ram_blobs", "disk_blobs"}
    row = s["recent"][-1]
    for k in ("rid", "t", "status", "prompt", "cached", "fed", "completion", "queue_s",
              "ttft_s", "prefill_s", "prefill_tps", "decode_tps", "e2e_s", "turns", "swaps",
              "phases", "io"):
        assert k in row, k
    assert row["status"] == "length" and row["completion"] == 5 and row["ttft_s"] > 0
    assert row["e2e_s"] >= row["ttft_s"] >= row["queue_s"] >= 0
    assert s["series"] and set(s["series"][-1]) == {"t", "dec", "pre", "run", "wait", "pos",
                                                    "cpu"}
    # the card: the fake BCU's governor die, its BMC's measured power, RUN
    c = s["card"]
    assert s["card_note"] is None and c["source"] == "card" and c["ok"]
    assert c["die_src"] == "governor" and abs(c["die_c"] - 62.0) < 0.2
    assert c["gov"]["state"] == "RUN" and c["power"]["source"] == "measured"
    assert c["reg"]["hot_c"] == 52.0 and c["bitstream"]["core_mhz"] == 150.0
    assert set(c["util"]) == set(TM.UNITS) | {"DRAM"}
    assert s["card_series"] and s["card_series"][-1]["die"] == c["die_c"]
    assert s["monitor"]["card_samples"] >= 2 and s["monitor"]["card_interval_s"] == 0.5
    json.dumps(s, allow_nan=False)          # (no NaN or Infinity: the browser's JSON.parse)


def test_the_program_cache_report(stack, monkeypatch, tmp_path):
    """Saves off and the wait entries computed: /v1/stats, the snapshot, /metrics, the page's pill.
    Over the floor it says saving; with no disk cache there is no gauge, never a made-up zero."""
    from opentpu import progcache as PC
    from opentpu import waits as WC
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path / "not" / "yet"))    # (its first parent's disk)
    monkeypatch.setattr(PC, "FREE_FLOOR", 1 << 62)
    monkeypatch.setitem(stack.mon.info, "warm", {"tokens": 51200, "of_tokens": 65536, "buckets": 200})
    stack.mon.tick()
    monkeypatch.setitem(WC.stats, "computed", WC.stats["computed"] + 328)
    monkeypatch.setitem(WC.stats, "seconds", WC.stats["seconds"] + 26.0)
    time.sleep(0.25)
    s = json.loads(_get(stack.port, "/api/stats")[2])
    pc = s["prog_cache"]
    assert pc["saving"] is False and pc["dir"].endswith("not/yet")
    assert pc["under_floor_bytes"] == (1 << 62) - pc["free_bytes"] > 0
    assert pc["entries_last_min"] == 328 and pc["entries_s_last_min"] == 26.0, pc
    assert s["server"]["warm"]["tokens"] == 51200
    v = json.loads(_get(stack.port, "/v1/stats")[2])["prog_cache"]
    assert v["saving"] is False and v["floor_bytes"] == 1 << 62
    kinds, samples = _parse(_get(stack.port, "/metrics")[2].decode())
    got = {n: x for n, _, x in samples}
    assert kinds["otpu:prog_cache_saving"] == "gauge" and got["otpu:prog_cache_saving"] == 0
    assert got["otpu:prog_cache_floor_bytes"] == 1 << 62 and got["otpu:prog_cache_free_bytes"] > 0
    assert kinds["otpu:wait_entries_computed_total"] == "counter"
    assert got["otpu:wait_entries_computed_total"] >= 328
    assert got["otpu:wait_entries_compute_seconds_total"] >= 26.0
    assert got["otpu:prog_cache_warm_tokens"] == 51200
    page = _get(stack.port, "/")[2].decode()
    assert 'id="saves"' in page and "program saves off" in page and 'id="warm"' in page
    monkeypatch.setattr(PC, "FREE_FLOOR", 0)
    stack.mon.tick()
    assert json.loads(_get(stack.port, "/api/stats")[2])["prog_cache"]["saving"] is True
    monkeypatch.setenv("OTPU_PROG_CACHE", "0")
    stack.mon.tick()
    pc = json.loads(_get(stack.port, "/api/stats")[2])["prog_cache"]
    assert pc["saving"] is None and pc["free_bytes"] is None
    assert "otpu:prog_cache_saving" not in stack.mon.prometheus()


def test_the_sampler_reads_smis_default_path_only(stack):
    """The governor's die, never the SYSMON's DRP registers (smi --sysmon), and no writes."""
    import opentpu.host.gov as G
    fk = FakeBcu(run=True, kick_ms=300, run_ms=754_000)
    seen = []
    read = fk.reg_read
    fk.reg_read = lambda off: seen.append(off) or read(off)
    cs = TM.CardSampler("/dev/fakebcu0", 2.0, open_transport=lambda dev: fk)
    cs.sample()
    cs.sample()
    assert not [o for o in seen if G.R_SYSMON <= o < G.R_SYSMON + 0x10000]
    assert fk.writes == [] and cs.samples == 2 and cs.cpu_s > 0
    # the governor's reading gone (TEMP_VALID clear): smi falls back to the SYSMON's register
    fk2 = FakeBcu(run=True, kick_ms=300, run_ms=754_000, gov_valid=False, die_c=66.0)
    c = TM.CardSampler("/dev/fakebcu0", 2.0, open_transport=lambda dev: fk2).sample()
    assert c["die_src"] != "governor" and abs(c["die_c"] - 66.0) < 0.6


def test_a_samples_span_is_its_overlap():
    """telemetry.overlap: the seconds of a window during which a card sample ran (a save's or a
    restore's "telemetry" part), none long before it."""
    fk = FakeBcu(run=True, kick_ms=300, run_ms=754_000)
    cs = TM.CardSampler("/dev/fakebcu0", 2.0, open_transport=lambda dev: fk)
    t0 = time.perf_counter()
    cs.sample()
    t1 = time.perf_counter()
    assert TM.overlap(t0, t1) > 0 and TM.overlap(t0 - 1e6, t0 - 1e6 + 1.0) == 0


# ---------------------------------------------------------------------------- /metrics
_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)\{([^}]*)\} (\S+)$')


def _parse(text):
    """{name: kind}, [(name, labels dict, value)] of a text exposition, checked line by line."""
    kinds, samples, helped = {}, [], set()
    assert text.endswith("\n")
    for line in text.splitlines():
        if line.startswith("# HELP "):
            helped.add(line.split()[2])
            continue
        if line.startswith("# TYPE "):
            _, _, name, kind = line.split()
            assert name in helped and name not in kinds, line
            kinds[name] = kind
            continue
        m = _SAMPLE.match(line)
        assert m, line
        name, lab, v = m.groups()
        labels = dict(re.findall(r'(\w+)="((?:[^"\\]|\\.)*)"', lab))
        val = float(v.replace("+Inf", "inf"))
        assert val == val, line                     # (no NaN)
        samples.append((name, labels, val))
    return kinds, samples


def _family(name, kinds):
    for suf in ("_bucket", "_sum", "_count"):
        if name.endswith(suf) and kinds.get(name[:-len(suf)]) == "histogram":
            return name[:-len(suf)]
    return name


def test_metrics_format(stack):
    _, _, body = _get(stack.port, "/metrics")
    kinds, samples = _parse(body.decode())
    for name, labels, v in samples:
        fam = _family(name, kinds)
        assert fam in kinds, name                   # (every sample after its TYPE)
        assert labels.get("model_name") == "tiny-qwen3", name
        if kinds[fam] == "counter":
            assert name.endswith("_total") and v >= 0, name
    assert kinds["vllm:generation_tokens_total"] == "counter"
    for c in ("otpu:prefix_entry_hits_total", "otpu:prefix_entry_hit_tokens_total",
              "otpu:prefill_device_runs_total"):
        assert kinds.get(c) == "counter", c         # (a prompt's prefixes kept: their restores)
    for g in ("vllm:num_requests_running", "vllm:num_requests_waiting",
              "vllm:kv_cache_usage_perc", "otpu:die_temp_celsius", "otpu:power_watts",
              "otpu:governor_run", "otpu:card_up", "otpu:utilization_ratio",
              "otpu:kv_positions_used", "otpu:kv_positions_capacity", "otpu:uptime_seconds"):
        assert kinds.get(g) == "gauge", g
    for h in ("vllm:time_to_first_token_seconds", "vllm:e2e_request_latency_seconds",
              "vllm:time_per_output_token_seconds", "vllm:request_queue_time_seconds",
              "vllm:request_prefill_time_seconds", "vllm:request_decode_time_seconds",
              "vllm:request_prompt_tokens", "vllm:request_generation_tokens"):
        assert kinds.get(h) == "histogram", h
        b = [(float(lab["le"].replace("+Inf", "inf")), v) for n, lab, v in samples
             if n == h + "_bucket"]
        assert b == sorted(b) and b[-1][0] == float("inf")      # (le rising, counts too)
        assert [v for _, v in b] == sorted(v for _, v in b)
        cnt = next(v for n, _, v in samples if n == h + "_count")
        assert b[-1][1] == cnt and cnt >= 1, h
    got = {(n, tuple(sorted(lab.items()))): v for n, lab, v in samples}
    ok = got[("vllm:request_success_total", (("finished_reason", "length"),
                                             ("model_name", "tiny-qwen3")))]
    assert ok >= 1
    assert got[("otpu:governor_run", (("model_name", "tiny-qwen3"),))] == 1
    assert got[("otpu:power_watts", (("model_name", "tiny-qwen3"), ("source", "measured")))] > 0
    assert {lab["unit"] for n, lab, _ in samples if n == "otpu:utilization_ratio"} == \
        set(TM.UNITS) | {"DRAM"}


def test_histogram_lines():
    h = M.Histogram((0.1, 1.0))
    for v in (0.05, 0.1, 0.5, 3.0):
        h.observe(v)
    assert h.lines("x", 'a="b"') == [
        'x_bucket{a="b",le="0.1"} 2', 'x_bucket{a="b",le="1.0"} 3',
        'x_bucket{a="b",le="+Inf"} 4', 'x_sum{a="b"} 3.65', 'x_count{a="b"} 4']


# ---------------------------------------------------------------------------- /api/events
def _events(port, path="/api/events", n=3, timeout=30, until=None):
    """The first n events of an /api/events stream (or those up to the first `until(event,
    data)` holds for): [(event, data)], and the retry line."""
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request("GET", path)
    r = c.getresponse()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    out, ev, retry = [], None, None
    while (len(out) < n) if until is None else not (out and until(*out[-1])):
        line = r.readline().decode()
        assert line, "the stream ended"
        line = line.rstrip("\n")
        if line.startswith("retry: "):
            retry = int(line[7:])
        elif line.startswith("event: "):
            ev = line[7:]
        elif line.startswith("data: "):
            out.append((ev, json.loads(line[6:])))
    c.close()
    return out, retry


def test_events_stream(stack):
    """`hello` (the snapshot with its series and the recent table), then a `tick` a monitor
    tick, their seq rising; a request that ends between ticks puts `recent` in the next."""
    out, retry = _events(stack.port, n=3)
    assert retry == 3000
    (e0, s0), (e1, s1), (e2, s2) = out
    assert e0 == "hello" and "series" in s0 and "recent" in s0 and "card_series" in s0
    assert e1 == e2 == "tick" and s0["seq"] < s1["seq"] < s2["seq"]
    assert "series" not in s1 and s1["point"]["t"] >= s0["point"]["t"]

    got = []

    def ended(e, s):
        return e == "tick" and any(x["completion"] == 3 for x in s.get("recent", ()))
    t = threading.Thread(target=lambda: got.append(_events(stack.port, timeout=120,
                                                           until=ended)[0]))
    t.start()
    time.sleep(0.5)
    st, _ = _call(stack.port, "POST", "/v1/completions", {"prompt": "count", "max_tokens": 3,
                                                          "temperature": 0})
    t.join(timeout=120)
    assert st == 200 and got and ended(*got[0][-1])
    assert all("recent" not in s for e, s in got[0][1:-1] if e == "tick")


# ---------------------------------------------------------------------------- the API key
def test_the_key(stack):
    """With --api-key: the page is open (it holds no data), the monitor's routes take the key as
    a Bearer header or ?key= (the browser's EventSource sets no header), the API only the
    header; the request log does not show the key."""
    p = stack.kport
    assert _get(p, "/")[0] == 200
    for path in ("/api/stats", "/metrics", "/v1/models"):
        assert _get(p, path)[0] == 401
        assert _get(p, path, {"Authorization": f"Bearer {KEY}"})[0] == 200
        assert _get(p, path, {"Authorization": "Bearer wrong"})[0] == 401
    assert _get(p, f"/api/stats?key={KEY}")[0] == 200
    assert _get(p, f"/metrics?key={KEY}")[0] == 200
    assert _get(p, "/api/stats?key=wrong")[0] == 401
    assert _get(p, f"/v1/models?key={KEY}")[0] == 401
    out, _ = _events(p, f"/api/events?history=1&key={KEY}", n=1)
    assert out[0][0] == "hello"
    line = API._KEY_RE.sub(r"\1-", f'"GET /api/events?key={KEY}&x=1 HTTP/1.1" 200 -')
    assert KEY not in line and "key=-&x=1" in line


# ---------------------------------------------------------------------------- the chat's SSE
def _raw_stream(port, body):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", "/v1/chat/completions", json.dumps({**body, "stream": True}),
              {"Content-Type": "application/json"})
    r = c.getresponse()
    assert r.status == 200 and r.getheader("Content-Type") == "text/event-stream"
    raw = r.read().decode()
    c.close()
    return raw


def test_chat_stream_is_openais_sse(stack):
    """Every event one `data: ` line and a blank line; chunks of object chat.completion.chunk
    with one id, created and model, a choice's index and delta; with include_usage the last
    chunk before [DONE] has no choices and the usage."""
    msgs = [{"role": "user", "content": "tell me"}]
    raw = _raw_stream(stack.port, {"messages": msgs, "max_tokens": 6, "temperature": 0,
                                   "stream_options": {"include_usage": True}})
    assert raw.endswith("data: [DONE]\n\n")
    events = [e for e in raw.split("\n\n") if e]
    assert all(e.startswith("data: ") and "\n" not in e for e in events)
    chunks = [json.loads(e[6:]) for e in events[:-1]]
    assert len({c["id"] for c in chunks}) == 1 and all(c["id"].startswith("chatcmpl-")
                                                       for c in chunks)
    for c in chunks:
        assert c["object"] == "chat.completion.chunk" and c["model"] == "tiny-qwen3"
        assert isinstance(c["created"], int)
    *body, last = chunks
    assert all(len(c["choices"]) == 1 and c["choices"][0]["index"] == 0 and "delta" in
               c["choices"][0] for c in body)
    assert body[-1]["choices"][0]["finish_reason"] == "length"
    assert last["choices"] == [] and last["usage"]["completion_tokens"] == 6
    u = last["usage"]
    assert u["total_tokens"] == u["prompt_tokens"] + u["completion_tokens"]
    # without include_usage: no usage chunk
    raw = _raw_stream(stack.port, {"messages": msgs, "max_tokens": 3, "temperature": 0})
    assert all(json.loads(e[6:]).get("choices") for e in raw.split("\n\n")[:-2] if e)


def test_the_openai_client(stack):
    """The openai package's client (when installed) over the API: models, a chat completion, a
    streamed one with include_usage, a text completion."""
    openai = pytest.importorskip("openai")
    cl = openai.OpenAI(base_url=f"http://127.0.0.1:{stack.kport}/v1", api_key=KEY)
    m = cl.models.list().data[0]
    assert m.id == "tiny-qwen3"
    msgs = [{"role": "user", "content": "hello"}]
    r = cl.chat.completions.create(model="tiny-qwen3", messages=msgs, max_tokens=5,
                                   temperature=0)
    text = r.choices[0].message.content
    assert r.usage.completion_tokens == 5 and r.choices[0].finish_reason == "length"
    st = cl.chat.completions.create(model="tiny-qwen3", messages=msgs, max_tokens=5,
                                    temperature=0, stream=True,
                                    stream_options={"include_usage": True})
    parts, usage = [], None
    for ch in st:
        if ch.choices:
            parts.append(ch.choices[0].delta.content or "")
        if ch.usage is not None:
            usage = ch.usage
    assert "".join(parts) == text and usage.completion_tokens == 5
    r = cl.completions.create(model="tiny-qwen3", prompt="abc", max_tokens=3, temperature=0)
    assert r.usage.completion_tokens == 3
    with pytest.raises(openai.AuthenticationError):
        openai.OpenAI(base_url=f"http://127.0.0.1:{stack.kport}/v1",
                      api_key="wrong").models.list()


# ---------------------------------------------------------------------------- n/a
def _monitor(card=None, note=None, eng=None):
    eng = eng or SimpleNamespace(pos=0, cap=CAP, batch=1)
    w = SimpleNamespace(eng=eng, stats={}, cv=threading.Condition(), running=[], waiting=[],
                        observers=[], usage=lambda: {})
    return M.Monitor(w, "m", card=card, card_note=note, interval=0.05, host=_NoHost())


def test_no_card_is_na():
    """No card (a simulator): card None and why; /metrics has none of the card's or the host's
    gauges (left out, not zeros)."""
    mon = _monitor(note="no card: the isa backend")
    s = mon.snapshot()
    assert s["card"] is None and s["card_note"] == "no card: the isa backend"
    assert s["card_point"] is None and s["kv"]["dram"] is None
    text = mon.prometheus()
    for g in ("otpu:card_up", "otpu:die_temp_celsius", "otpu:power_watts", "otpu:governor_run",
              "otpu:utilization_ratio", "otpu:host_cpu_ratio", "otpu:host_memory_used_bytes",
              "otpu:decode_tokens_per_second"):
        assert g not in text, g
    _parse(text)
    assert "None" not in text and "nan" not in text.lower()


def test_another_card_has_no_governor_or_bmc():
    """A card without the BCU's governor and BMC (FakeTransport): its XADC die, no regulator,
    no governor, no power (no Vivado report for its build) -- each n/a, and out of /metrics."""
    ft = FakeTransport()
    cs = TM.CardSampler("/dev/fake0", 2.0, open_transport=lambda dev: ft)
    mon = _monitor(card=cs)
    cs.sample()
    s = mon.tick()
    c = s["card"]
    assert c["die_src"] == "XADC" and c["die_c"] is not None
    assert c["gov"] is None and c["reg"] is None and c["power"] is None
    text = mon.prometheus()
    _parse(text)
    assert "otpu:die_temp_celsius" in text and "otpu:utilization_ratio" in text
    for g in ("otpu:power_watts", "otpu:governor_run", "otpu:regulator_temp_celsius"):
        assert g not in text, g


def test_a_safe_card_and_a_failing_one():
    """The accelerator fenced (SAFE): the governor's die and the BMC's power still, no
    utilization, card_up 0. A transport that fails: no card, its error in the note, the
    sampler's thread alive and counting failures."""
    fs = FakeBcu(run=False)
    cs = TM.CardSampler("/dev/fakebcu0", 2.0, open_transport=lambda dev: fs)
    mon = _monitor(card=cs)
    cs.sample()
    c = mon.tick()["card"]
    assert not c["ok"] and c["gov"]["state"] == "SAFE" and c["util"] is None
    assert c["die_src"] == "governor" and c["power"]["w"] > 0
    text = mon.prometheus()
    assert 'otpu:card_up{model_name="m"} 0' in text and "otpu:utilization_ratio" not in text
    assert 'otpu:governor_run{model_name="m"} 0' in text

    def gone(dev):
        raise OSError(19, "No such device", dev)
    bad = TM.CardSampler("/dev/xdma9", 0.5, open_transport=gone)
    mon = _monitor(card=bad)
    bad.interval = 0.01
    bad.start()
    t = time.monotonic()
    while bad.failures < 3 and time.monotonic() - t < 10:
        time.sleep(0.01)
    s = mon.tick()
    bad.stop()
    assert bad.failures >= 3 and bad.samples == 0 and bad.thread.is_alive() is False
    assert s["card"] is None and "No such device" in s["card_note"]
    assert "No such device" in s["card_error"]
    assert "otpu:card_up" not in mon.prometheus()


# ---------------------------------------------------------------------------- the card's panels
def _page_units():
    """The utilization panels' keys: what each one's sparkline reads off a card point (the
    page's UNITS)."""
    m = re.search(r"const UNITS = \[(.*?)\];", API.DASHBOARD.read_text(), re.S)
    return re.findall(r'\["(\w+)", "', m.group(1))


def test_the_utilization_panels_read_the_cards_keys():
    """The card's counters as smi names them (RUNNING, MXU_BUSY, MXU_MAC, VPU_BUSY, QNT_BUSY,
    DMA_BUSY, DRAM) through compact and the monitor's history: a number under every key the
    page's panels read, from a sample shaped as the card's and from smi's query of the fake
    BCU; the sampler's spacing and duration in the monitor's block."""
    keys = _page_units()
    assert keys == ["run", "mxu", "mac", "vpu", "dma", "dram"]
    real = {"RUNNING": 0.91, "MXU_BUSY": 0.57, "MXU_MAC": 0.41, "VPU_BUSY": 0.22,
            "QNT_BUSY": 0.04, "DMA_BUSY": 0.18, "DRAM": 0.83}
    mon = _monitor()
    mon._card_sample(TM.compact({"time": 100.0, "ok": True, "util": real, "dram_gbs": 61.2}))
    pt = mon.card_series[-1]
    names = ("RUNNING", "MXU_BUSY", "MXU_MAC", "VPU_BUSY", "DMA_BUSY", "DRAM")
    assert {k: pt[k] for k in keys} == {k: real[n] for k, n in zip(keys, names)}
    fs = FakeBcu(run=True)
    cs = TM.CardSampler("/dev/fakebcu0", 2.0, open_transport=lambda dev: fs)
    mon = _monitor(card=cs)
    cs.on_sample(cs.sample())                 # (as the sampler's thread does)
    time.sleep(0.05)
    cs.on_sample(cs.sample())
    s = mon.tick()
    hist = mon.snapshot(history=True)["card_series"]
    assert len(hist) == 2 and hist[-1] == s["card_point"]
    for p in hist:
        assert all(isinstance(p[k], float) for k in keys), p
    mo = s["monitor"]
    assert mo["card_took_s"] >= 0 and mo["card_took_max_s"] >= mo["card_took_s"]
    assert mo["card_spacing_s"] >= 0.05 and mo["card_spacing_max_s"] >= mo["card_spacing_s"]


def test_the_governors_figures():
    """Trip and resume to 0.1 C (the fake's codes read 79.87003988749996 and 55.00119191875001);
    the thermal trips (TRIPS [15:0]) apart from the heartbeat's ([31:16]) and from STATUS's
    causes: a host SAFE is a cause, never a trip."""
    fs = FakeBcu(run=False, cause=0x8, trips=(1, 2))
    cs = TM.CardSampler("/dev/fakebcu0", 2.0, open_transport=lambda dev: fs)
    mon = _monitor(card=cs)
    cs.sample()
    g = mon.tick()["card"]["gov"]
    assert (g["trip_c"], g["resume_c"], g["reg_trip_c"]) == (79.9, 55.0, 78.0)
    assert g["trips_thermal"] == 1 and g["trips_heartbeat"] == 2
    assert g["causes"] == ["host SAFE"] and g["state"] == "SAFE"
    page = API.DASHBOARD.read_text()
    assert 'fnum(g.trip_c, 1)' in page and '"last stop" : "stopped by"' in page


def test_power_is_labelled_unverified():
    """The BCU's power is the LTC3884's READ_PIN (the watch's W_PIN): measured, but a lower
    bound of VCCINT's input never checked against a meter (verified False), the Vivado
    estimate beside it when there is one; an estimate alone says so."""
    c = TM.compact({"time": 1.0, "ok": True,
                    "measured": {"w": 4.3, "source": "watch", "how": "PIN", "vin": 11.98,
                                 "label": "VCCINT regulator input", "age_ms": 40},
                    "power": {"w": 13.4}})
    assert c["power"] == {"w": 4.3, "source": "measured", "label": "VCCINT regulator input",
                          "age_ms": 40, "vin": 11.98, "how": "PIN", "verified": False,
                          "reading": "LTC3884 READ_PIN", "estimate_w": 13.4}
    c = TM.compact({"time": 1.0, "ok": True, "power": {"w": 13.4}})
    assert c["power"]["source"] == "estimate" and c["power"]["verified"] is False
    assert c["power"]["reading"] is None and c["power"]["estimate_w"] is None
    fs = FakeBcu(run=True)
    cs = TM.CardSampler("/dev/fakebcu0", 2.0, open_transport=lambda dev: fs)
    mon = _monitor(card=cs)
    cs.sample()
    assert mon.tick()["card"]["power"]["reading"] == "LTC3884 READ_PIN"
    help_ = [ln for ln in mon.prometheus().splitlines() if ln.startswith("# HELP otpu:power_watts")]
    assert help_ and "lower bound" in help_[0] and "not checked against a meter" in help_[0]
    assert "unverified" in API.DASHBOARD.read_text()


def test_the_page_ships_in_the_wheel():
    """pyproject's package-data has the dashboard page (an installed otpu-serve serves it)."""
    import tomllib
    from pathlib import Path
    pp = tomllib.loads((Path(API.__file__).parents[2] / "pyproject.toml").read_text())
    data = pp["tool"]["setuptools"]["package-data"]["opentpu"]
    assert "serve/dashboard.html" in data


# ---------------------------------------------------------------------------- not in the way
class _Stuck:
    """A transport whose register reads block until released (a sample stuck on the bus)."""
    def __init__(self, fk):
        self.fk, self.gate, self.reads = fk, threading.Event(), 0

    def __getattr__(self, n):
        return getattr(self.fk, n)

    def reg_read(self, off):
        self.reads += 1
        self.gate.wait()
        return self.fk.reg_read(off)

    def reg_read_many(self, offs):
        return [self.reg_read(o) for o in offs]


def test_a_stuck_sample_does_not_stall_requests(stack):
    """The card's sampler blocked inside a register read: a chat completion, /api/stats,
    /metrics and the monitor's ticks go on; the snapshot says it waits for the first sample.
    Released, the sample lands."""
    stuck = _Stuck(FakeBcu(run=True, kick_ms=300, run_ms=754_000))
    cs = TM.CardSampler("/dev/fakebcu0", 0.5, open_transport=lambda dev: stuck)
    mon = M.Monitor(stack.w, "tiny-qwen3", card=cs, interval=0.05)
    srv = API.Server(stack.w, stack.tok, stack.spec, "tiny-qwen3", stack.defaults, monitor=mon)
    h = _serve(srv)
    port = h.server_address[1]
    mon.start()
    cs.start()
    try:
        t = time.monotonic()
        while not stuck.reads and time.monotonic() - t < 10:
            time.sleep(0.01)
        assert stuck.reads == 1                 # (the sampler's thread inside its first read)
        seq0 = mon.seq
        t0 = time.monotonic()
        st, r = _call(port, "POST", "/v1/chat/completions",
                      {"messages": [{"role": "user", "content": "still there?"}],
                       "max_tokens": 4, "temperature": 0})
        assert st == 200 and r["usage"]["completion_tokens"] == 4
        st, _, body = _get(port, "/api/stats")
        s = json.loads(body)
        assert st == 200 and s["card"] is None
        assert s["card_note"] == "waiting for the first sample"
        assert _get(port, "/metrics")[0] == 200
        assert mon.seq > seq0 and time.monotonic() - t0 < 120
        assert stuck.reads == 1 and cs.samples == 0
    finally:
        stuck.gate.set()
    t = time.monotonic()
    while cs.samples < 1 and time.monotonic() - t < 10:
        time.sleep(0.01)
    assert cs.samples >= 1 and mon.tick()["card"]["gov"]["state"] == "RUN"
    h.shutdown()
    h.server_close()
    mon.stop()
    stack.w.observers.remove(mon._ended)


# ---------------------------------------------------------------------------- in flight
def test_a_prompts_progress_and_eta():
    """While a prompt is fed (Worker.feeding): its tokens done are the device's position past
    its start, the rate theirs over its time so far, the ETA what is left at that rate; before
    the first part lands, the ETA from the recent prompts' rate."""
    eng = SimpleNamespace(pos=1000, cap=262144, batch=1)
    mon = _monitor(eng=eng)
    w = mon.w
    r = SC.Request(list(range(9000)), max_new=64)
    r.t_submit, r.t_start = time.perf_counter() - 12.0, time.perf_counter() - 10.0
    w.running = [r]
    w.stats = {"prefill_tokens": 500}
    w.feeding = (r, 1000, 8000, time.perf_counter() - 10.0, 500)
    mon.recent_q.append({"prefill_tps": 40.0, "ttft_s": 1.0, "e2e_s": 2.0, "decode_tps": 30.0})
    s = mon.tick()
    inf = s["inflight"]
    assert inf["phase"] == "prefill" and inf["done"] == 0 and inf["of"] == 8000
    assert inf["rate_tps"] is None and inf["eta_from"] == "recent prompts"
    assert inf["eta_s"] == pytest.approx(8000 / 40.0, rel=0.01)
    eng.pos = 1000 + 2000
    s = mon.tick()
    inf = s["inflight"]
    assert inf["done"] == 2000 and inf["eta_from"] == "this prompt"
    assert inf["rate_tps"] == pytest.approx(200.0, rel=0.05)
    assert inf["eta_s"] == pytest.approx(6000 / inf["rate_tps"], rel=0.02)
    assert s["throughput"]["prefill_tps"] == inf["rate_tps"]
    assert s["throughput"]["fed"] == 2500 and s["queue"]["items"][0]["state"] == "prefill"
    assert s["server"]["state"] == "prefill"
    text = mon.prometheus()
    assert 'otpu:prompt_progress_tokens{model_name="m"} 2000' in text
    assert 'otpu:prompt_remaining_tokens{model_name="m"} 6000' in text


# ---------------------------------------------------------------------------- --demo
def test_the_demo():
    """otpu-serve --demo: 30 minutes of history, a simulated card marked demo, requests that
    run; the chat routes answer 503."""
    from opentpu.serve import demo as D
    srv, w, mon = D.open_demo(minutes=2, interval=0.05, card_interval=0.05,
                              first=(600, 0, 20, 0.2))
    h = _serve(srv)
    try:
        port = h.server_address[1]
        s = json.loads(_get(port, "/api/stats?history=1")[2])
        assert s["server"]["demo"] is True and len(s["series"]) >= 100
        t = time.monotonic()
        while (s["card"] is None or not s["recent"]) and time.monotonic() - t < 20:
            time.sleep(0.1)
            s = json.loads(_get(port, "/api/stats")[2])
        assert s["card"]["source"] == "demo" and s["recent"]
        assert s["kv"]["dram"] and s["kv"]["cap"] == D.CAP
        st, r = _call(port, "POST", "/v1/chat/completions",
                      {"messages": [{"role": "user", "content": "x"}]})
        assert st == 503 and r["error"]["type"] == "demo"
        _parse(_get(port, "/metrics")[2].decode())
    finally:
        h.shutdown()
        h.server_close()
        mon.stop()
        w.stop()


# ---------------------------------------------------------------------------- the fans
def _post_gains(port, body, ctype="application/json", key=None):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    h = {"Content-Type": ctype}
    if key:
        h["Authorization"] = f"Bearer {key}"
    c.request("POST", "/api/fan_gains", json.dumps(body), h)
    r = c.getresponse()
    out = json.loads(r.read() or b"{}")
    c.close()
    return r.status, out


def _fan_status(tmp_path, name, fan_no, card, t):
    (tmp_path / name).write_text(json.dumps({
        "time": t, "interval_s": 2.0, "die_c": 60.5, "reg_c": 66.0, "card": card, "fan_no": fan_no,
        "fan": {"duty": 70, "why": None, "mode": "pid", "target_c": 60.0, "filt_c": 60.2,
                "lead": "vreg", "gains": {"kp": 4.0, "ki": 0.1, "kd": 0.0}, "i": 5.5},
        "fan_pid": {"version": "1.4"}}))


def test_the_fans_state_and_gains(stack, tmp_path, monkeypatch):
    """fan_pid's status files on /api/stats, age_s how old they are; none: fans None. The route:
    one fan's gains into the file fan_pid reads, the other's as they were, every bad value
    refused with nothing written, JSON's content type and the key required."""
    from opentpu.serve import fans as FN
    monkeypatch.setenv("OTPU_FAN_DIR", str(tmp_path))
    assert FN.state() is None
    t = time.time()
    _fan_status(tmp_path, "status.json", 1, "card0 fan 1", t - 3)
    _fan_status(tmp_path, "status2.json", 2, "card1 fan 2", t - 500)
    (tmp_path / "status3.json").write_text("{}")
    f = FN.state(now=t)
    assert [(x["fan"], x["card"], x["age_s"]) for x in f] == [(1, "card0 fan 1", 3.0),
                                                              (2, "card1 fan 2", 500.0)]
    assert f[0]["gains"] == {"kp": 4.0, "ki": 0.1, "kd": 0.0} and f[0]["i"] == 5.5
    (tmp_path / "status2.json").write_text("not json")
    assert [x["fan"] for x in FN.state(now=t)] == [1]       # (a bad one: only that fan's gone)
    stack.mon.tick()
    assert _call(stack.port, "GET", "/api/stats")[1]["fans"][0]["card"] == "card0 fan 1"

    live = tmp_path / FN.LIVE_GAINS
    st, r = _post_gains(stack.port, {"fan": 1, "kp": 4, "ki": 0.25, "kd": 0})
    assert st == 200 and r == {"fan": 1, "gains": {"kp": 4.0, "ki": 0.25, "kd": 0.0}}
    st, r = _post_gains(stack.port, {"fan": 2, "kp": 5, "ki": 0, "kd": 1.5})
    assert st == 200
    want = {"fan1": {"kp": 4.0, "ki": 0.25, "kd": 0.0}, "fan2": {"kp": 5.0, "ki": 0.0, "kd": 1.5}}
    assert json.loads(live.read_text()) == want
    assert [p.name for p in tmp_path.glob("*.tmp")] == []
    # the target: taken, kept by a later body without one, the other fan's and its gains as they were
    st, r = _post_gains(stack.port, {"fan": 2, "kp": 5, "ki": 0, "kd": 1.5, "target_c": 65})
    assert st == 200 and r["gains"]["target_c"] == 65.0
    st, r = _post_gains(stack.port, {"fan": 2, "kp": 6, "ki": 0, "kd": 1.5})
    assert st == 200 and r["gains"] == {"kp": 6.0, "ki": 0.0, "kd": 1.5, "target_c": 65.0}
    want["fan2"] = r["gains"]
    assert json.loads(live.read_text()) == want
    for t in (39.9, 70.1, "65", True, None):
        assert _post_gains(stack.port, {"fan": 2, "kp": 1, "ki": 1, "kd": 1, "target_c": t})[0] == 400, t
    assert json.loads(live.read_text()) == want
    for bad in ({"fan": 3, "kp": 1, "ki": 1, "kd": 1}, {"fan": True, "kp": 1, "ki": 1, "kd": 1},
                {"fan": [1], "kp": 1, "ki": 1, "kd": 1}, {"kp": 1, "ki": 1, "kd": 1},
                {"fan": 1, "kp": -1, "ki": 1, "kd": 1}, {"fan": 1, "kp": 101, "ki": 1, "kd": 1},
                {"fan": 1, "kp": 1, "ki": 11, "kd": 1}, {"fan": 1, "kp": 1, "ki": 1, "kd": 1001},
                {"fan": 1, "kp": "1", "ki": 1, "kd": 1}, {"fan": 1, "kp": None, "ki": 1, "kd": 1},
                {"fan": 1, "kp": True, "ki": 1, "kd": 1}, {"fan": 1, "kp": 1, "ki": 1}):
        assert _post_gains(stack.port, bad)[0] == 400, bad
    c = http.client.HTTPConnection("127.0.0.1", stack.port, timeout=60)
    c.request("POST", "/api/fan_gains", '{"fan": 1, "kp": NaN, "ki": 1, "kd": 1}',
              {"Content-Type": "application/json"})
    assert c.getresponse().status == 400
    c.close()
    assert _post_gains(stack.port, {"fan": 1, "kp": 1, "ki": 1, "kd": 1}, ctype="text/plain")[0] == 415
    g = {"fan": 1, "kp": 1, "ki": 1, "kd": 1}
    assert _post_gains(stack.kport, g)[0] == 401
    assert _post_gains(stack.kport, g, key="wrong")[0] == 401
    assert json.loads(live.read_text()) == want
    assert _post_gains(stack.kport, g, key=KEY)[0] == 200
    stack.mon.fans, keep = None, stack.mon.fans
    try:
        assert _post_gains(stack.port, g)[0] == 404         # (the demo's)
    finally:
        stack.mon.fans = keep


def test_each_cards_temperatures_on_the_series():
    """Cards in tensor parallel: each one's die and regulator on the card series' point beside
    the hottest; one card: only the hottest."""
    mon = M.Monitor(SimpleNamespace(observers=[]), "m", interval=1.0, host=_NoHost(), fans=None)
    one = {"t": 1.0, "die_c": 61.0, "reg": {"hot_c": 66.0}}
    mon._card_sample(one)
    assert "die0" not in mon.card_series[-1]
    mon._card_sample(dict(one, cards=[{"name": "card0", "die_c": 61.0, "reg": {"hot_c": 66.0}},
                                      {"name": "card1", "die_c": 59.5, "reg": None},
                                      {"name": "x", "ok": False}]))
    p = mon.card_series[-1]
    assert (p["die"], p["die0"], p["reg0"], p["die1"], p["reg1"], p["die2"], p["reg2"]) == \
        (61.0, 61.0, 66.0, 59.5, None, None, None)

#!/usr/bin/env python3
"""x2 card qualification: identical platform-7 images, both cables, small collectives,
the co-sim's Qwen3-0.6B tokens, and opt-in prebuilt configuration-only performance cuts.
Checkpoint and cut packs are read only; expected results come from --expect / --cuts,
never from a model reference computed on the card host.

Every completed launch records the two hubs' sequences and completion checks plus all link
counters. X2_FRAMES counts received frames, not errors: framing / length / overflow failures
are included in X2_CRC and the port stickies. No automatic retry or link reset hides a fault.
Real devices are re-armed under two register-only thermal guards (no JTAG), then left SAFE.
Run as a child of the caller's otpu-lock hold, preserving OTPU_LOCK_HELD / OTPU_LOCK_PID.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import replace
import json
import os
from pathlib import Path
import signal
import statistics
import sys
import time

sys.dont_write_bytecode = True

import numpy as np

from opentpu import target as T
from opentpu.host import gov as G, regs as R
from opentpu.host.board import Board, BoardBackend, device_config, open_transport
from opentpu.host.cards import CardsTransport, card_info
from tools.x2_session.safety import SessionGuard

# the checkpoint (read only: X2_MODEL) and, below, the directory a real card run's report must be under (X2_RUNS)
MODEL = Path(os.environ.get("X2_MODEL") or Path(__file__).resolve().parents[1] / "models/Qwen3-0.6B")


def runs_dir() -> Path:
    return Path(os.environ.get("X2_RUNS") or Path.home() / "otpu-build/bcu-wg/runs")


class QualError(RuntimeError):
    pass


def quiet_environment():
    """All optional disk caches disabled. Thread limits keep image building modest; the
    model's int8 image is built once, with no ISA model run or tokenizer download."""
    os.environ.update(OTPU_IMG_CACHE="0", OTPU_IMAGE_CACHE="0", OTPU_PROG_CACHE="0",
                      OTPU_BUILD_JOBS="1", PYTHONDONTWRITEBYTECODE="1",
                      OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")


def power_up(cards, out: Path, stack: ExitStack):
    """wg_bringup leaves SAFE at exit. Keep both governors alive for the combined run,
    using their own die and BMC watch registers, without reading the SYSMON DRP window."""
    guards = []
    for c, card in enumerate(cards):
        g = G.Gov(card)
        if not g.present():
            raise QualError(f"card {c}: no thermal governor")
        stack.callback(g.safe)
        s = g.status()
        if s["temp_c"] >= 80 or (s.get("reg_c") or 0) >= 80:
            raise QualError(f"card {c}: temperature >= 80 C")
        g.safe()
        g.set_limits(trip_c=80, resume_c=65, reg_trip_c=78)
        guard = SessionGuard(card, csv=out.parent / f"{out.stem}-thermal-{c}.csv")
        stack.callback(guard.stop)
        guard.read_first()
        guard.start()
        guards.append(guard)
        deadline = time.monotonic() + 1800
        while True:
            if guard.tripped or time.monotonic() > deadline:
                raise QualError(f"card {c}: GO refused: {guard.tripped or g.status()}")
            if g.go(dimms=()):
                break
            time.sleep(2)
        g.set_dimms((0, 1, 2, 3), acc=False)
        time.sleep(5)
        if guard.tripped or not g.status()["run"]:
            raise QualError(f"card {c}: guard during calibration: {guard.tripped or 'SAFE'}")
        g.set_dimms((0, 1, 2, 3), acc=True)
        card.poll(R.R_ID, 0xFFFFFFFF, R.ID_OTPU, 10)
    return guards


def snapshot(ct):
    result = []
    for c, card in enumerate(ct.cards):
        ports = []
        for p in range(2):
            offs = [R.R_X2_STATUS[p], R.R_X2_HELLO[p], R.R_X2_CRC[p], R.R_X2_FRAMES[p],
                    R.R_X2_PING[p], *range(R.R_X2_COUNTERS[p], R.R_X2_COUNTERS[p] + 16, 4)]
            vals = card.reg_read_many(offs)
            ports.append(dict(zip(("status", "hello", "crc_errors", "frames_received", "ping",
                                   "tx_words", "rx_words", "credit_wait", "soft_errors"), vals)))
            ports[-1]["ping_core_cycles"] = vals[4] & 0x7FFFFFFF if vals[4] >> 31 else None
        info, seq, coll = card.reg_read_many([R.R_X2_INFO, R.R_X2_SEQ, R.R_COLL_STATUS])
        result.append({"card": c, "info": info, "seq": seq & 0xFFFF,
                       "coll_status": coll, "ports": ports})
    return result


def clean_links(snap):
    for card in snap:
        c = card["card"]
        if card["info"] != (1 | 32 << 8 | 2048 << 16):
            raise QualError(f"card {c}: unsupported X2_INFO {card['info']:#x}")
        if card["coll_status"] & R.COLL_ERRORS:
            raise QualError(f"card {c}: COLL_STATUS {card['coll_status']:#x}")
        for p, port in enumerate(card["ports"]):
            if port["crc_errors"] or port["soft_errors"] or port["status"] & R.X2_ERRORS \
                    or not port["status"] & R.X2_UP:
                raise QualError(f"card {c} port {p}: link error {port}")


def micro_programs(kind):
    """test_cards' three-BAR loop, and test_isa's LD / row-strided GATHER / ST pattern,
    widened to four slices and 256 columns (32 payload words per piece on both cables)."""
    from opentpu import isa as I
    if kind == "bar":
        return [[I.loop(1, 3), I.bar(), I.halt()]] * 4, 3, 0
    p = [I.ld(0x2000, 0, 512), I.gather(0, 4096, 2, 256, 256, 1024, 256),
         I.st(0x4000, 4096, 2048), I.halt()]
    return [p] * 4, 1, 8192


def hardware_engine(cfg, ct):
    from opentpu.llm.qwen3 import Engine, Spec, load_weights
    return Engine(Spec.from_hf(MODEL), load_weights(MODEL), cap=256, cfg=cfg,
                  wformat="int8", resident=True, prog_cache=False,
                  backend=lambda c, imgs: BoardBackend(c, imgs, transport=ct, status=False))


def expectation(path):
    case = json.loads(path.read_text())["qwen3_0_6b"]
    for key in ("prompt", "next_tokens", "generated"):
        v = case[key]
        if not isinstance(v, list) or not v or any(type(x) is not int or x < 0 for x in v):
            raise QualError(f"--expect: {key} must be a nonempty list of token IDs")
    if len(case["prompt"]) != len(case["next_tokens"]):
        raise QualError("--expect: prompt / next_tokens length mismatch")
    if len(case["prompt"]) + len(case["generated"]) > 256:
        raise QualError("--expect: case exceeds the 256-token capacity")
    return case


def cut_projection(meta, stats, arrays, cfg):
    """Use a complete card-0 trace when available; explicitly label span-fraction fallback.
    The production trace is 16384 records, smaller than q38/goss's original cut traces.
    A truncated trace cannot supply full-token phase spans, but does not invalidate the
    independent memory/count checks or the measured whole-cut cycle counter."""
    from tools.x2_cosim import phase_spans, project
    cycles, cosim = stats["cycles"], meta["cosim"]
    tr = stats.get("trace")
    trace = {k: v for k, v in (tr or {}).items() if k != "records"}
    # A keep-first buffer stops TRACE_COUNT at its depth on the card, so lost reads 0 once it
    # fills: a full buffer is truncated too (q38-4's card-0 trace at 16384, 2026-11-03).
    full = bool(tr) and tr.get("keep") == "first" and tr.get("count", 0) >= tr.get("depth", 1 << 62)
    spans, why = None, "trace incomplete" if tr else "no trace"
    if cycles and tr and not tr["lost"] and not tr["drop"] and not full:
        from opentpu import isa as I
        from opentpu.host.hwlens import drop_orphans, records_to_trace
        from opentpu.profile import parse
        programs = [[I.Instr.decode(w[i:i + 8]) for i in range(0, len(w), 8)]
                    for w in [arrays[f"program_{k}"] for k in range(4)]]
        p = parse(drop_orphans(records_to_trace(tr["records"], programs)), cfg, programs, "card")
        spans = phase_spans(p, blocks={int(k): v for k, v in meta["pc_phases"].items()})
        source = "card-0 trace"
        if not spans.get("HEAD"):
            spans, why = None, "trace has no HEAD span"
    if spans is None:
        spans = {k: v * cycles / cosim["cycles"] for k, v in cosim["phase_cycles"].items()}
        source = f"co-sim span fractions ({why})"
    out = {"cosim_cycles": cosim["cycles"], "one_card_cycles": cosim["one_card_cycles"],
           "trace": trace, "phase_source": source, "phase_cycles": spans,
           "delta_cosim_percent": 100 * (cycles / cosim["cycles"] - 1) if cycles else None,
           "one_card_speedup": cosim["one_card_cycles"] / cycles if cycles else None,
           "projection": project(meta["model"], cycles, spans) if cycles else None}
    if out["projection"]:
        out["projection"]["useful_gb_s_150mhz"] = 153.6 * out["projection"]["useful_bandwidth_fraction"]
    return out


def run_cuts(board, ct, cfg, packs, reps, run_checked, record, thermal):
    from tools.x2_cutpack import CHUNK, load_pack, low_sha
    info = board.info()
    khz = [t.reg_read(R.R_CORE_KHZ) for t in ct.cards]
    if len(set(khz)) != 1:
        raise QualError(f"step f: core clocks differ: {khz} kHz")
    for path in packs:
        meta, arrays = load_pack(path)
        name, low = meta["name"], meta["low_end"]
        if meta["machine_hash"] != info["cfg"]["hash"] or any(
                meta["card_hash"] != i["cfg"]["hash"] for i in ct.infos):
            raise QualError(f"{name}: pack ARCH_HASH does not match cards")
        if not 0 < low <= cfg.DRAM_BYTES or not low <= meta["program_at"] < min(cfg.DRAM_BYTES, 1 << 32):
            raise QualError(f"{name}: invalid pack memory bounds")
        ranges = meta["zero_ranges"]
        if not ranges or ranges[0][0] != 0 or ranges[0][1] < low or any(
                not 0 <= a < b <= cfg.DRAM_BYTES for a, b in ranges):
            raise QualError(f"{name}: invalid pack zero ranges")
        words = [arrays[f"program_{k}"] for k in range(4)]
        entries = [arrays[f"waits_{k}"] for k in range(4)]
        if any(w.dtype != np.dtype("<u4") or w.ndim != 1 or len(w) % 8 or
               len(w) // 8 > cfg.IMEM_WORDS for w in words):
            raise QualError(f"{name}: invalid pack programs")
        issued = meta["collectives_mod65536"]
        if not 0 <= issued <= 0xFFFF:
            raise QualError(f"{name}: invalid pack collective count")
        # Retain the manifest identity with every run, including the source co-sim and its
        # st16 comparison. The lead stages these directories read-only; no pack is modified.
        record("f", {"run": name, "prepare": True, "pack": str(path),
                     "pack_manifest": meta, "core_khz": khz})
        cycles = []
        zeros = np.zeros(CHUNK, np.uint8)
        for rep in range(reps):
            thermal()
            begin = time.monotonic()
            # Every repetition starts from the reference's initial DRAM, including all high
            # weights and low state. This also clears previous cuts and arbitrary junk.
            for k in range(4):
                for a, b in ranges:
                    for off in range(a, b, CHUNK):
                        thermal()
                        board.write(off, zeros[:min(CHUNK, b - off)], k)
                for p in meta["present"]:
                    board.write(p["addr"], arrays[p["array"]], k)
            zero_seconds = time.monotonic() - begin
            # No Board.load, wait pass, compiler, image builder or ISA reference on the host.
            board.load_program(meta["program_at"], words, entries)
            board.set_args(meta["args"] + [0] * (8 - len(meta["args"])))
            ct.collectives = issued             # includes dynamic LOOP counts, modulo 65536
            run = f"{name}-{rep + 1}"
            try:
                stats = run_checked(run, lambda: board.run(
                    trace={"keep": "first"} if info["caps"]["trace"] else None), issued)
            finally:
                if board.in_run:
                    board.abort(0)
            if stats["instructions"] != meta["instructions"]:
                raise QualError(f"{run}: instruction counts {stats['instructions']} != {meta['instructions']}")
            io, vloc = meta["logits_addr"], meta["logits_words_per_slice"]
            logits = []
            for k in range(4):
                got = board.read(io + 4 * k * vloc, 4 * vloc, k).view("<u4")
                if not np.array_equal(got, arrays[f"logits_{k}"]):
                    raise QualError(f"{run}: logits mismatch on slice {k}")
                logits.append(got)
            token = int(np.argmax(np.concatenate(logits).view("<f4")))
            if token != meta["argmax"]:
                raise QualError(f"{run}: argmax {token} != {meta['argmax']}")
            hashes = []
            check_start = time.monotonic()
            for k in range(4):
                got = low_sha(lambda a, n, k=k: board.read(a, n, k), low, thermal)
                if got != meta["low_sha256"][k]:
                    raise QualError(f"{run}: low-area sha256 mismatch on slice {k}")
                hashes.append(got)
            thermal()
            nzero = 4 * sum(b - a for a, b in ranges)
            result = {"run": run, "rep": rep + 1, "cycles": stats["cycles"],
                      "instructions": stats["instructions"], "argmax": token,
                      "low_sha256": hashes, "checked_low_bytes_per_slice": low,
                      "zero_bytes": nzero, "initialize_seconds": zero_seconds,
                      "initialize_gb_s": nzero / zero_seconds / 1e9,
                      "verify_seconds": time.monotonic() - check_start,
                      **cut_projection(meta, stats, arrays, cfg)}
            cycles.append(stats["cycles"])
            record("f", result)
        record("f", {"run": name, "summary": True, "repetitions": reps, "cycles": cycles,
                     "min_cycles": min(cycles), "median_cycles": statistics.median(cycles),
                     "cosim_cycles": meta["cosim"]["cycles"],
                     "one_card_cycles": meta["cosim"]["one_card_cycles"]})


def qualify(cards, target, steps, expected, report, save, watchdog, engine_factory=None,
            cfg=None, guards=(), backends=None, cuts=(), cut_reps=3):
    def record(name, data):
        report["steps"].append({"step": name, **data})
        save()
        print(f"{name}: {json.dumps(data)}", flush=True)

    def thermal():
        for c, guard in enumerate(guards):
            if guard.tripped or not guard.g.status()["run"]:
                raise QualError(f"card {c}: thermal guard: {guard.tripped or 'SAFE'}")

    # Always run a; b is a prerequisite for c/d/f. --steps selects work, never skips safety.
    identities = [dict(card_info(t), core_khz=t.reg_read(R.R_CORE_KHZ)) for t in cards]
    record("a", {"cards": [dict(i, arch_hash=i["cfg"]["hash"], platform=i["cfg"]["platform"])
                            for i in identities]})
    ct = CardsTransport(cards, target=target, link=False, watchdog=watchdog)
    if (ct.ncards, ct.spc, ct.slices) != (2, 2, 4):
        raise QualError("expected two cards with two slices each")
    if any(i["cfg"]["platform"] != 7 or not i["xcard"] for i in ct.infos):
        raise QualError("expected PLATFORM 7 and CAPS2 XCARD on both cards")
    board = Board(ct, lock=False)
    info = board.info()              # exact target / ARCH_HASH check on real devices too
    cfg = cfg or device_config(info)
    if not set(steps) & {"b", "c", "d", "f"}:
        return
    ct.link = True
    ct.set_cards()                   # checks stopped cores; topology, LRST, hello, watchdog
    for card in cards:
        card.reg_write(R.R_X2_CTRL, R.X2_HELLO)
    cable_map = [{"card": c, "port": p, "peer_card": d, "peer_port": q}
                 for (c, p), (d, q) in sorted(ct.links().items())]
    for c, card in enumerate(cards):
        card.reg_write(R.R_X2_CTRL, R.X2_PING)
        for p, off in enumerate(R.R_X2_PING):
            try:
                card.poll(off, 1 << 31, 1 << 31, 2)
            except TimeoutError:
                raise QualError(f"card {c} port {p}: ping timed out") from None
    snap = snapshot(ct)
    record("b", {"cable_map": cable_map, "links": snap})
    clean_links(snap)
    thermal()

    def completed(name, before, expected_delta=None):
        # Board.wait / BoardBackend already check every HALTED poll. Also check explicitly,
        # and retain our raw baseline: later register flushes must not erase the evidence.
        completion_delta = ct.collective_delta
        failure = None
        try:
            ct.check_run()
        except Exception as e:
            failure = str(e)
        snap = snapshot(ct)
        delta = [(x["seq"] - b) & 0xFFFF for x, b in zip(snap, before)]
        record("e", {"run": name, "before_seq": before, "delta": delta,
                     "completion_delta": completion_delta, "links": snap,
                     "completion_error": failure})
        if failure:
            raise QualError(failure)
        clean_links(snap)
        if len(set(delta)) != 1 or (not delta[0] and expected_delta is None) or \
                (expected_delta is not None and delta != [expected_delta] * 2):
            raise QualError(f"{name}: X2_SEQ delta {delta}, expected {expected_delta}")
        thermal()

    def sequences():
        thermal()
        return [t.reg_read(R.R_X2_SEQ) & 0xFFFF for t in cards]

    def run_checked(name, fn, count=None):
        before = sequences()
        try:
            return fn()
        finally:
            completed(name, before, count)

    if "c" in steps:
        from opentpu.isasim import Machine
        for kind in ("bar", "gather"):
            programs, count, nbytes = micro_programs(kind)
            rcfg = replace(cfg, DRAM_BYTES=1 << 20, BHW=0)
            ref = Machine(rcfg, programs, [None] * 4)
            rng = np.random.default_rng(2)
            for s in range(4):
                data = rng.integers(0, 1 << 32, 512, dtype=np.uint32).view(np.uint8)
                board.write(0x2000, data, s)
                ref.slices[s].dram[0x2000:0x2800] = data
            ref.run()
            board.load(0x10000, programs)
            stats = run_checked(kind, board.run, count)
            if stats["instructions"] != [s.icount for s in ref.slices]:
                raise QualError(f"{kind}: instruction counts differ from the ISA reference")
            for s in range(4):
                got = board.read(0x4000, nbytes, s) if nbytes else np.array([], np.uint8)
                if not np.array_equal(got, ref.slices[s].dram[0x4000:0x4000 + nbytes]):
                    raise QualError(f"{kind}: gathered data mismatch on slice {s}")
            record("c", {"run": kind, "checked_bytes_per_slice": nbytes,
                         "instructions": stats["instructions"], "cycles": stats["cycles"]})
    if "d" in steps:
        record("d", {"run": "prepare", "model": str(MODEL), "wformat": "int8",
                     "cap": 256, "resident": True})
        engine = (engine_factory or hardware_engine)(cfg, ct)
        if backends is not None:
            backends.append(engine.backend)
        try:
            next_tokens = []
            for pos, tok in enumerate(expected["prompt"]):
                logits = run_checked(f"decode-{pos}", lambda: engine.step(tok))
                got = int(np.argmax(logits))
                next_tokens.append(got)
                record("d", {"run": f"decode-{pos}", "input": tok, "next_token": got,
                             "cycles": engine.stats[-1]["cycles"],
                             "instructions": engine.stats[-1]["instructions"]})
                if got != expected["next_tokens"][pos]:
                    raise QualError(f"decode-{pos}: token mismatch: {got} != "
                                    f"{expected['next_tokens'][pos]}")
            got = run_checked("generate", lambda: engine.generate_card(
                next_tokens[-1], len(expected["generated"]), stop_ids=[]))
            record("d", {"run": "generate", "generated": got,
                         "cycles": engine.stats[-1]["cycles"],
                         "instructions": engine.stats[-1]["instructions"]})
            if got != expected["generated"]:
                raise QualError(f"generate: token mismatch: {got} != {expected['generated']}")
        finally:
            if backends is None:
                engine.backend.close()
    if "f" in steps:
        try:
            run_cuts(board, ct, cfg, cuts, cut_reps, run_checked, record, thermal)
        finally:
            board.close()


def main(argv=None, *, cards=None, engine_factory=None, cfg=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dev", required=True, help="/dev/xdmaA+/dev/xdmaB, card 0 first")
    ap.add_argument("--target", default="bcu1525-x2:st16")
    ap.add_argument("--expect", type=Path)
    ap.add_argument("--json", type=Path, required=True)
    ap.add_argument("--steps", default="a,b,c,d")
    ap.add_argument("--cuts", help="comma-separated read-only cut-pack directories; implies f")
    ap.add_argument("--cut-reps", type=int, default=3)
    ap.add_argument("--coll-wd", type=lambda x: int(x, 0), default=R.COLL_WD_DEFAULT)
    a = ap.parse_args(argv)
    steps = a.steps.split(",")
    if a.cuts and "f" not in steps:
        steps.append("f")
    if not steps or any(s not in "abcdf" or len(s) != 1 for s in steps):
        ap.error("--steps: comma-separated a,b,c,d,f (e runs after every launch)")
    cuts = [Path(p) for p in a.cuts.split(",")] if a.cuts else []
    if "f" in steps and not cuts:
        ap.error("step f requires --cuts PACK[,PACK...]")
    if a.cut_reps < 1:
        ap.error("--cut-reps: positive integer")
    if not 0 <= a.coll_wd <= 0xFFFFFFFF:
        ap.error("--coll-wd: uint32 core cycles, 0 disables")
    devs = a.dev.split("+")
    if len(devs) != 2 or len(set(devs)) != 2 or any(not d.startswith("/dev/xdma") for d in devs):
        ap.error("--dev: two distinct /dev/xdma paths separated by +")
    if cards is None and not a.json.resolve().is_relative_to(runs_dir().resolve()):
        ap.error(f"real card output must be under {runs_dir()}/ (X2_RUNS)")
    quiet_environment()
    a.json.parent.mkdir(parents=True, exist_ok=True)
    report = {"ok": False, "dev": devs, "target": a.target, "requested_steps": steps,
              "steps": [], "model": str(MODEL), "coll_wd": a.coll_wd,
              "cuts": [str(p) for p in cuts], "cut_reps": a.cut_reps}

    def save():
        a.json.write_text(json.dumps(report, indent=2) + "\n")

    def interrupted(sig, frame):
        raise QualError(f"signal {sig}: session stopped")

    old = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        expected = expectation(a.expect) if "d" in steps and a.expect else None
        if "d" in steps and expected is None:
            raise QualError("step d requires --expect FILE")
        target = T.load(a.target)
        with ExitStack() as stack:
            guards, backends = [], []

            def close_backends():
                for backend in backends:
                    backend.close()

            if cards is None:
                cards = []
                for d in devs:
                    t = open_transport(d)
                    stack.callback(t.close)
                    cards.append(t)
                # Backend.close releases CardsTransport's device locks. Register it before
                # the guards, so both governors reach SAFE while those locks are still held.
                stack.callback(close_backends)
                guards = power_up(cards, a.json, stack)
            else:
                stack.callback(close_backends)
            # Stop cores before closing DMA or the guards, including exceptions mid-run.
            def stop():
                # Clear Board.in_run and drain writes while accelerator registers still
                # answer. Backend.close later releases locks, after both governors are SAFE.
                for backend in backends:
                    backend.board.abort(0)
                for t in cards:
                    t.reg_write(R.R_CTRL, 0)
            stack.callback(stop)
            extra = {"cuts": cuts, "cut_reps": a.cut_reps} if "f" in steps else {}
            qualify(cards, target, steps, expected, report, save, a.coll_wd,
                    engine_factory, cfg, guards, backends, **extra)
        report["ok"] = True
        save()
        print("PASS x2 card qualification", flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as e:
        report["error"] = " ".join(str(e).splitlines())
        save()
        print(f"FAIL x2: {report['error']}", file=sys.stderr, flush=True)
        return 1
    finally:
        for s, handler in old.items():
            signal.signal(s, handler)


if __name__ == "__main__":
    raise SystemExit(main())

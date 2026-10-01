#!/bin/bash
# offload: Qwen3.5-35B-A3B's streaming on a PCIe Gen2 image (ld-memch's session, the image as
# loaded: no reload; docs/offload.md 6). slot_bench (the 35B's split pool warm: ms an expert,
# the DMA thread's GB/s), then session 7's runs in its order: 128 tokens, decayed use, slots
# filling the DRAM, the first 16 tokens against q35ref16: q35e128a/b (the embedding table on the
# host, 42 slots a layer) and q35c128a/b (on the card, 34), against session 7's Gen1 (O/s7).
# Needs a tree at or after main 2a0b962 (the DMA guard). About 10 min.
# Run: otpu-lock --wait 10800 -- tools/offload/sessions/gen2.sh   (log: O/gen2/session.log)
set -u
SESSION=gen2; source "$(dirname "$0")/env.sh"
exec > >(tee -a $R/session.log) 2>&1
echo "gen2 start $(date +%T) tree $rev mem $(mem) GB"
echo "--- card: the 35B's split pool, warm"
timeout 600 python tools/offload/slot_bench.py --reqs 40 --sizes q35 --pool $O/pool-q35-fp4.split.bin --warm 2>&1 | tail -1
RUNS="${RUNS:-q35e128a q35c128a q35e128b q35c128b}" bash "$here/card_moe.sh"
echo "gen2 end $(date +%T)"
date > $R/DONE

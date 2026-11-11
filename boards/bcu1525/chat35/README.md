# Qwen3.6-35B-A3B chat on the BCU1525, its whole context

`otpu-chat --model qwen36-35b` on the card host runs `otpu-chat35`: one card (`bcu1525-wg:bhw16`) brought up under
the guard, the 35B resident in its 32 GiB at all 262,144 positions (fp4 experts, nvfp4 attention / DeltaNet / shared
MLP, int8 head, the generate loop on the card), the card back in SAFE when the chat ends.

## The host's file: BCU_ENV

Nothing here names a card or a host. The wrappers and `boards/bcu1525/scripts/fan_pid.py` read a shell file,
`BCU_ENV` (default `~/.config/otpu/bcu.env`), or the same names from the environment:

```sh
BCU_CARD0_SERIAL=...        # the card's JTAG cable serial (hw_server's target), required
BCU_CARD0_BDF=0000:BB:00.0  # its PCI function, required
BCU_CARD0_NAME=card0        # the name in messages and on the dashboard (default cardN)
BCU_CARD1_SERIAL=...        # a second card, for the x2 wrappers
BCU_CARD1_BDF=0000:BB:00.0
BCU_BUILD=~/otpu-build      # optional: the trees, cardimg, bin (default ~/otpu-build)
BCU_DATA=$BCU_BUILD/data    # optional: raw images and KV stores
BCU_PY=python3              # optional: the interpreter (default PATH's python3)
BCU_CARD=0                  # optional: the one-card wrappers' card (0 or 1; `otpu-chat --card N` sets it)
BCU_IMAGE=one               # optional: the image the cards hold (one, or x2: the two-card image) when a card does not say
BCU_RESUME=65               # optional: the governor's RESUME for every bring-up's GOs (C, at most 69.9)
BCU_TREE=$BCU_BUILD/bcu-wg/tree   # optional: otpu-chat's host tree
BCU_X2_TARGET=bcu1525-x2:st16     # optional: the two-card image's target, for otpu-chat on both cards
```

A `BCU_*` variable already in the environment wins over the file's, so one run can differ
(`BCU_BUILD=~/other otpu-chat`, `BCU_RESUME=69.9 otpu-chat35x2`).

**RESUME.** The governor takes a GO only with the die at or under RESUME, and a bring-up sets RESUME before its GOs
(`wg_bringup.py up --resume`, 65 C by default). Fans that hold the die above it (a fan loop whose target is higher)
keep every GO refused: the bring-up retries for 30 minutes and fails. `BCU_RESUME` raises it for the wrappers'
bring-ups. Each wrapper says so before the bring-up when the die is above the RESUME it will use, and while it waits
it shows the bring-up's line (`waiting for GO: die 70.1 C, RESUME 65.0 C ...`) every 15 s.

**Which image.** The card says (`python3 -m opentpu.host.cardimage --dev DEV`; `otpu-smi`'s Image rows show the same:
target, slices, BUILD_ID, CFG_HASH, the core clock, and where the answer came from). In RUN its accelerator's own
registers answer. In SAFE they are fenced, and the answer is the governor's identity words on a build that has them
(`otpu_bcu_gov`'s IMAGE registers, 0x50-0x60), else what the last bring-up since the boot read from the running card
(a record under `$BCU_BUILD/state/`, kept only for the boot it was written in: a new image needs a JTAG load and a
warm reboot). A card that says nothing (a build without the words that no host has brought up since the boot) leaves
it to `BCU_IMAGE`, else `x2` when `$BCU_BUILD/cardimg/x2/IMAGE` exists, else `one`; a card's answer wins over
`BCU_IMAGE`, and `otpu-chat` says so when they differ. On the two-card image `otpu-chat`
brings both cards up (`tools/x2_session/bringup.py up` on each) and chats on both
(`--dev /dev/xdmaA+/dev/xdmaB`, `OTPU_TARGET=$BCU_X2_TARGET`), any model the chat takes; `--card 0` or `1` there is
refused at once, since one card of that image holds half the machine's slices. On a one-card image `--card 0|1`
picks the card, as before. A bring-up that then fails on the bitstream's check says the card holds another image
than `BCU_IMAGE`. The two-card image's target is the one the card names (its CFG_HASH's variant) unless
`BCU_X2_TARGET` is set.

A card's XDMA device is found by its PCI function each time, so either card runs the one-card wrappers whichever
`/dev/xdmaN` it is (`BCU_DEV`, when set, must name that card's device: a check, not a choice). `OTPU_DEV` is not part of
an image's or a program's key, so one built image serves both cards.

Each wrapper's `T` (its tree, `$B/tree-TREE`) and `ENTRY` (its image's key, `KEY`) are placeholders: installing a
wrapper means setting those two lines to the tree it runs from and the key `c35.sh key` (or `c35x2.sh key`) prints
for that tree.

The names start with `BCU_` because the wrappers drop every `OTPU_*` of the caller's (the image key's environment
stays the entry's). A wrapper without its card's serial and PCI function stops before any card step.

## Pieces

| File | Where it runs | What it does |
|---|---|---|
| `otpu-chat` | card host `$BCU_BUILD/bin` | the BCU chat wrapper: one card of a one-card image (`--card 0|1`) or both cards of the two-card image (no `--card`, or `--card x2`); `--model qwen36-35b` goes to `otpu-chat35` or `otpu-chat35x2` |
| `otpu-chat35` | card host `$BCU_BUILD/bin` | bring-up, the raw image's pages into the page cache meanwhile, `chat35.py`, SAFE |
| `chat35.py` | card host / build host, from the tree's root | the chat with the entry's Engine (see below); `CHAT35_BUILD=1` builds the image, `CHAT35_KEY=1` prints its key, `CHAT35_EXPAND=<entry>` writes the raw image |
| `c35.sh`, `b35.sh` | build host | key / build the image there (88 min, ~45 GB RAM: on a build host, not the card host); `[TREE [ARGS...]]`: another tree, otpu-chat's ARGS (`--slots 2 --cap 131072`: otpu-serve35 --slots 2's image) |
| `warm35.py`, `w35.sh` | build host | every bucket's programs into `qprog35` (1,024 buckets, 12 processes, 26 min): the generate loop's (greedy and the chat's sampler) with their wait entries, the resident decode step with its |
| `chk35.py` | build host / card host | one bucket's decode step in the cache (key, load, wait entries) |
| `otpu-serve35` | card host `$BCU_BUILD/bin` | the same bring-up, then `serve35.py`: the 35B over an OpenAI-style API with the dashboard (below); SAFE on Ctrl-C / SIGTERM |
| `serve35.py` | card host `$BCU_BUILD/cardimg` | otpu-serve (`opentpu/serve/api.py`) over `chat35.open_engine()`: the same engine, the layer runs warmed while the queue is empty |
| `serve35_check.py` | anywhere that reaches the port | the first card check of a fresh server: the routes, a context saved and written back, the same turn fresh after `/flush_cache` (the same greedy text), the card's telemetry, `/metrics` |
| `serve35_batch_check.py` | anywhere that reaches the port | otpu-serve35 `--slots N`'s check (batched decode, `opentpu/serve/batching35.py`): three prompts alone (greedy references), then two and three at once, each reply its reference's word for word; then two sampled with production's sampling and a seed each, alone and at once, each reply its alone's; per-request and total tok/s, the step timing |
| `otpu-serve35x2`, `serve35x2.py` | card host `$BCU_BUILD/bin`, `cardimg/x2` | the same server on both BCU1525s (`bcu1525-x2:st16`): both bring-ups, `serve35x2.py` over chat35x2's engine (below; on the cards: docs/batch35.md section 4) |

## What chat35.py changes in otpu-chat (host side only: the image key stays the entry's)

- The Engine is built as the entry was: the pickled target configuration, `--cap 262144`, the family's formats
  (streamed.FORMATS), resident, no prompt runs, not the streamed-experts path; a cache miss is refused.
- The image: the entry as a sparse raw file on the SSD, mapped (`raw_images`), not read through the cache's runs. Its
  data extents (SEEK_DATA / SEEK_HOLE) are the Images' `runs`: the scrub leaves them and only they are written
  (Board, boardload); `CHAT35_RUNS=0` writes the spans.
- The checkpoint opens at its first tensor read (`_LazyWeights`): on a cache hit nothing reads one (the rows come from
  the image's lookup tables), so neither torch nor the 26 safetensors headers on the HDD load (-11 s).
- A prompt part of `CHAT35_LM_MIN` (64) tokens or more goes layer by layer, `CHAT35_LM` (2) rows a run
  (`Engine(layer_major=2)`, prefill_layers); shorter parts through the decode step a token a run. After each turn the
  engine's compile workers build the next two buckets' layer runs (the first runs in a bucket waited ~3.4 s for them).
- `CHAT35_TIMES=1`: the load's and decode's host calls timed to stderr; `CHAT35_PROF=<file>`: a cProfile.

## Serving it: otpu-serve35

```sh
otpu-serve35                                   # in tmux on the card host: http://127.0.0.1:8000/v1, the dashboard at /
SERVE35_API_KEY=KEY otpu-serve35 --port 8001   # a key (the dashboard: http://127.0.0.1:8001/#key=KEY)
ssh -L 8000:127.0.0.1:8000 HOST                # from another machine, then http://127.0.0.1:8000/ in a browser
python3 serve35_check.py                       # a fresh server's check (ALL PASS; 2-4 min)
```

`otpu-serve35` checks the card and its governor, refuses while another card job runs (`otpu-chat35` refuses while it
runs), brings the card up and holds RUN (`SERVE35_HOLD` minutes, default 30 days), loads the 35B through
`chat35.open_engine()` (the chat's setup, the same image key) and serves `/v1/chat/completions` (plain and SSE, with
`stream_options.include_usage`), `/v1/completions`, `/v1/models` (`max_model_len` 262144), `/health`, `/metrics`
(Prometheus, vLLM's names) and the dashboard (docs/dashboard.md: throughput, TTFT per request, the prompt in flight
with its ETA, the queue, the card's utilization, die and regulator temperature, power, the KV context, the card's
DRAM map and the KV store). One sequence: requests run in turns, each context saved to host RAM and written back for
its next turn; a chat's next turn feeds only its new tokens. A reply's start is a snapshot on the card (as the chat's):
the same chat's next turn rolls back to it with no save, and the worker saves it to host RAM when it goes idle
(docs/serve.md "In turns"). The card's telemetry is otpu-smi's default reads every
2 s on a thread of its own. Ctrl-C (or SIGTERM, or the hold's end, or a guard trip) stops the server (a run in
flight ends first, `SERVE35_GRACE` s at most) and then puts the card back in SAFE. The bring-up runs in a session
of its own in every BCU wrapper (otpu-serve35, otpu-chat35, otpu-chat; wg_bringup installs its own
SIGINT handler, so ignoring SIGINT is not enough): the terminal's Ctrl-C and hangup reach the server or the chat, never
the hold under them; the wrapper ends the hold after them and says SAFE from wg.json's `status_safe`
(`tests/test_chat35_wrappers.py`: the order with stub bring-ups). `chat35.py`'s command line and
environment are unchanged (`main()` and `open_engine()` share `install()`).

## Two cards: otpu-serve35x2

`otpu-serve35x2` serves the 35B on both BCU1525s as one machine, over the two-card chat's engine (`chat35x2.py`: four slices,
two a card, `CHAT35_DEV=/dev/xdmaA+/dev/xdmaB`). What is in place, each with a test
without a card (`tests/test_serve35x2.py`):

- The wrapper: `otpu-chat35x2`'s checks (both devices by PCI function, no card job, the fans' tmux, both governors),
  the raw image's prefetch, each card's bring-up (`tools/x2_session/bringup.py up`, RUN held `SERVE35_HOLD`) in
  a session of its own, the server as `otpu-serve35`'s (`SERVE35_GRACE`). Ctrl-C, SIGTERM or a hangup stops the
  server, then both bring-ups, then says each card's SAFE from its own `wg.json`; either card's hold ending stops
  the server and the other card. `TREE` and `KEY` are placeholders, as in `otpu-chat35x2`: a deploy fills both.
- `serve35x2.py`: `serve35.main` with the engine, the two devices, the cards' names and its label (`serve35.main`'s
  new arguments; `otpu-serve35`'s command line and behaviour unchanged). The engine: `chat35x2.open_engine()` when
  it has one, else `chat35x2.main()` over a stand-in for otpu-chat's Chat (chat35's way before `install()`).
  No idle layer warm-up (at S > 1 the layer runs come from the program cache: warm35x2.py).
- The cards' telemetry (`telemetry.CardsSampler`, the monitor's for a `+` device): a CardSampler a card (its own
  register transport, smi's default reads, its own BMC bridge under that device's BMC lock), the two sampled one
  after the other every 2 s, merged as one machine: the hottest die (its card named), RUN only when both governors
  are, the power and the DRAM traffic summed, the units' mean busy; the dashboard's card panel lists each card. A
  card that fails leaves the merged values n/a (never the other card's alone) and its error by name.
- The contexts over `CardsTransport` (kvio unchanged): each slice's regions at the same addresses on its card, the
  staging copies one program a slice on all four, read slice by slice. On IsaCards at S = 4 a context's page and
  state saved through the area equal the direct reads, and written back after another context give the ISA
  simulator's logits and greedy steps bit for bit (the test runs with `opentpu.host.cards`; skipped
  without it).

On the cards: docs/batch35.md section 4 (four slots, every reply against its reply alone) and docs/x2.md
section 3 (a run that fails on the link, and the recovery).

## Measured on the card (2026-11-03)

| | token steps (t35b) | layer-major R=2 + warm (t35d) |
|---|---|---|
| load (bring-up excluded; before boardload) | 30.6 s (scrub 12.2 s) | 29.1 s (scrub 12.3 s) |
| short turn TTFT | 1.18 / 1.54 s | 1.14 / 1.61 s |
| decode | 33.28 tok/s (1,024 tokens, no stall) | 33.11 tok/s |
| 2,689-token paste TTFT | 88.7 s (30.3 tok/s) | 65.7 s (40.9 tok/s, device 43.0) |
| 83-90 token follow-up TTFT | 3.20 s | 2.25 s |

Greedy replies are identical with token steps and layer-major runs (t35b against t35c). Decode is 4.5 Mcycles a token
at 150 MHz. A layer-major run costs ~3.5 Mcycles a token: the saving is the LM head the prompt's rows skip (~1.1 M),
not the row pairing; the prompt is DeltaNet-bound.

**The load (t35g):** 14.1 s against 29.6 s. The scrub leaves the image's 257 data runs
(19.27 GB, from SEEK_DATA / SEEK_HOLE on the raw file) and clears the rest, 6.2 s against 12.6 s; then only the runs are
written, split on 4 threads straight into the channel buffers. `tools/h2c_split_bench.py --gib 2` on the card host: the split
1.17 -> 5.88 GB/s at 8x128 with free DMA, 2.81 GB/s against a 2.9 GB/s link (link-bound). The greedy replies of t35c's
turns are the same, word for word. `CHAT35_RUNS=0` writes the spans as before.

**Long context (t35e, 2026-11-03):** a 30,870-token ledger (360 paragraphs) and two questions. The needle (paragraph
211: the password, the seal's hiding place) and the count and paragraph 0's town are all right.

| | t35e |
|---|---|
| 30,870-token prompt | 1,033 s, 29.9 tok/s (device 31.1) |
| decode at 31k | 19.6 tok/s (device 20.4), 7.37 Mcycles a token |
| 159-token follow-up at 31k | TTFT 12.9 s (device 6.6 s of it) |
| 90-token follow-up at 31k (t35h, the warm timed) | TTFT 4.1 s; prefill_layers 3.7 s = the device's; the warm of buckets 121-122 4.9 s |

Decode at depth costs ~9.3 cycles a position an attention layer (31k: +2.87 Mcycles). The MXU's arithmetic fits it
(D 128 x MCOLS 8, 8 query heads a KV head, head_dim 256: 2 KV heads x (2 K passes + 2 V^T passes) = 8 cycles at full
use), but that is a hypothesis until the board model's unit busy at depth says so (the VPU's
softmax or the quantizer could bound it instead). At 262,144 positions decode would be ~25 Mcycles a token (~6 tok/s).

## After a keyed change

Any edit under `opentpu/` but `host/` and `serve/` changes the image key: build the image (`b35.sh`), copy the entry
to the card host and expand it (`CHAT35_EXPAND`), `warm35` (`w35.sh 12`) and copy `qprog35`, point `otpu-chat35`'s and `otpu-serve35`'s `T`
and `ENTRY` at the new tree and key. A `host/` change is a copy of the file into the tree.

## Both cards: `otpu-chat35x2` (bcu1525-x2:st16)

The same chat on two cards as one machine: four slices, two a card, in tensor parallel over the QSFP28 link
(`opentpu.host.cards.CardsTransport`, `--dev /dev/xdmaA+/dev/xdmaB`, `BCU_CARD0_*` card 0). Both cards hold the x2 image (`tools/x2_card.py`,
`tools/x2_session`); `otpu-chat --model qwen36-35b` goes to
`otpu-chat35x2` when the cards hold the two-card image (`BCU_IMAGE=x2`, or `cardimg/x2/IMAGE` exists; or with
`--card x2`), else to `otpu-chat35`.

| File | Where it runs | What it does |
|---|---|---|
| `otpu-chat35x2` | card host `$BCU_BUILD/bin` | both bring-ups at once (`bringup.py up`: the session guard, RUN held), the raw image into the page cache meanwhile, `chat35x2.py`, both cards SAFE at the end |
| `chat35x2.py` | card host / build host, from the tree's root | chat35.py's chat with the x2 target's configuration (checked against the cards), `CHAT35_DEV`'s two cards, a turn whose run fails on the link asked again (`CardsTransport.recover`, the context back to its snapshot) |
| `c35x2.sh`, `b35x2.sh` | build host | key / build the image (`OTPU_TARGET` alone in the key: no `OTPU_DEV`): 57 min, peak RSS 34 GB, a 20.1 GB entry (`chat35x2.LayerExperts`: a layer's fused expert tensors converted once, not once an expert a slice, 8 h before); `[TREE [ARGS...]]`: another tree, otpu-chat's ARGS (`--slots 2 --cap 131072`: otpu-serve35x2 --slots 2's image) |
| `warm35x2.py`, `w35x2.sh [N] [TREE] [LB]` | build host | warm35's programs for the x2 image (`warm35x2.pkl`, a chat's; `AHEAD=1`: written first with no card, `chat35x2.py` `CHAT35_AHEAD`), with LB buckets 1 .. LB's first, then their layer-major runs, then the other buckets' |
| `purge35x2.py [PKL] [--dry]` | build host, card host | every bucket's resident decode step out of a program cache (the chat's keyed environment, from the tree's root): an earlier warm35.py cached slice 0's words alone at S > 1, which the cards ran with HALT on slices 1-3 (the first prompt token step never ended); w35x2.sh then warms them again |
| `cmp_replies.py A.out B.out` | card host | two `--plain` chat sessions' replies word for word (the timing lines out, and the lines starting `otpu-chat35: ` / `otpu-chat35x2: `, the wrappers' own, the closing SAFE line after the last prompt too: no model text goes); exit 1 on a mismatch |
| `snapdiff.py A B` | anywhere | two snapshots of the slices' KV caches and DeltaNet states (a card's against the ISA simulator's), compared region by region |
| `turns35.txt` | card host | a test session's four turns (the measurements' below) |

What changed under it: the generate loop gathers the slices' stop words at S > 1 (each slice read its
own, and the host's writes reach the cards one after another: a slice halted while the others ran into a collective);
layer-major runs at S > 1 (a program a slice, the MoE a row at a time through the tensor-parallel step's `moe_ffn`; at
S > 1 there are no compile workers, so the runs go through the program cache); the scrub and the image write go to both
cards at once (a thread a card: the slower link no longer holds the other card). `tests/test_chat_x2.py`: a tiny
Qwen3.5-MoE shaped as the 35B at S = 4 on `IsaCards`, bit for bit against the ISA simulator at S = 4 and S = 1.

**Measured on both cards (2026-11-04, the 150 MHz image on both).** x2a is t35c's
four turns (`turns35.txt`); the greedy replies are t35c's word for word, all four (`cmp_replies.py`: ALL SAME).

| | both cards (x2a) | one card (t35c) |
|---|---|---|
| load (bring-up excluded) | 160.6 s (the backends' start 126.0 s, scrub 17.9 s of it) | 32.5 s (scrub 12.6 s) |
| short turn TTFT (24 tokens) | 1.89 s | 1.15 s |
| decode (1,024 tokens) | 65.13 tok/s (device 65.3), 2.30 Mcycles a token | 33.27 tok/s (device 33.3), 4.51 Mcycles |
| 2,689-token paste TTFT | 57.37 s (device 97.6 tok/s) | 70.14 s (device 43.0 tok/s) |
| 90-token follow-up TTFT (2,749 reused) | 4.93 s (device 93.0 tok/s) | 8.57 s (device 41.3 tok/s) |

Decode is 1.96x and the paste's device rate 2.27x one card's. The paste's TTFT gains less (1.22x): the time outside the
device's runs grew from ~7.6 s (70.1 - 2,689 / 43.0) to ~29.8 s (57.4 - 2,689 / 97.6), not broken down on this session (at S > 1
the layer-major runs come through the program cache, with no compile workers); a later session with
`opentpu.host.runwords` (below) read 46.5 s for the same paste. The load is
5x one card's (the image's read from a hard disk, below); the write itself goes to both cards at once.

Two bugs came out on the cards first (both fixed, `tests/test_chat_x2.py` fails without each):
the first window hung on the first prompt token (warm35 had cached slice 0's decode words alone at S > 1: the other
three slices ran HALT; `BoardBackend.start` now refuses fewer programs than slices, `purge35x2.py` + `w35x2.sh` warmed
them again), and the second turned to garbage at position 256, the first HALT CHAIN into the next generate bucket's
slot (its wait entries kept FOLLOW, which the cross-card hub advances on local writes only: `Board.cross_card_entries`
clears it for `load_program` and the chain slots alike).

**Long context on both cards (2026-11-04: x2e = t35e's 30,870-token ledger and two questions): one word's
difference from one card, open.** The prompt and the first reply's first 7 words are t35e's; at word 8 (decode token ~9, position ~30,880,
inside one generate bucket: no chain) x2e says "the details regarding the vault and the guild's seal" where t35e says
"the answers to your questions", and both go on right (the password, the seal's place, [211]; 360 paragraphs,
Ravensby). The session run twice on the two cards gave the same replies word for word, so it is not a race between
the cards; a tiny model shaped as the 35B along 30,870 tokens gives the same logits bit for bit on the ISA simulator
at S = 1, at S = 4, through the cards' host path (`IsaCards`) and on the two cards, so it is not the split's
arithmetic in general. It has not been localized (docs/x2.md section 8).

| | both cards (x2e) | one card (t35e) |
|---|---|---|
| 30,870-token prompt | TTFT 668.5 s (device 78.5 tok/s) | 1,033.7 s (device 31.1 tok/s) |
| decode at 31k | 36.3 tok/s (device 48.4), 3.10 Mcycles a token | 19.6 tok/s (device 20.4), 7.37 Mcycles |
| follow-up at 31k (reused 30,866) | TTFT 3.68 s (99 tokens) | 12.94 s (159 tokens) |

x2e's TTFT outside the device (668.5 - 30,870 / 78.5 = ~275 s): the runs of a new bucket, 4,989 layer runs read from
the program cache (`_layer_run` 67 s: the cache's reader decodes each slice's words into instructions) and 5,047
`BoardBackend.start` calls 117 s (it assembles them again, then the wait entries and the load), the rest the per-run
path of ~617k runs. `opentpu.host.runwords` (installed by `chat35x2.py` with `CHAT35_LM`) hands the cache's words to
the board as they are: no decode, no assembly, the same programs word for word (`tests/test_chat_x2.py`: the logits
and the next step bit-equal to the compiling and the caching engines'). It changes no key; a later session with it read 564 s for x2e's first token, the replies unchanged.

The load is the HDD's: after the scrub (17.9 s) the image's 19.6 GB of data runs come from a rotational disk in 108 s, ~181 MB/s; one card's raw image is on the SSD (14 s).

## LFM2-2.6B on two cards

`CHAT35_MODEL=lfm2-2.6b` runs chat35x2.py and serve35x2.py on LFM2-2.6B, the 35B's entry points otherwise (the default,
unset, stays the 35B). It is dense, 30 layers, with one sequence, so no `--slots` and no batching. Its prompt goes as
run-time prompt runs (docs/prefill.md, `prefill.chunks`), not layer by layer (`CHAT35_LM=0`), in every bucket: at S > 1
there are no compile workers, so `prefill.covers`' other route would compile each run on the chat's thread
(`chat35x2.dense_prompts`).

The warm is `warm35x2.py N`: from `warm35x2.pkl` (`CHAT35_AHEAD=1 chat35x2.py`), buckets 1 .. 500's generate programs
(greedy and the chat's sampler, key (56, True)), their decode steps and wait entries, then every bucket's prompt runs
(`warm_prompts`: the fit, P and L at every size up to the bucket's R, wait entries; bucket 0: a context's first runs).
`CHAT35_NOEVICT=1` skips the cache scan on each write. Keep it complete: under the program cache's disk floor (docs/dashboard.md "Program saves") a server cannot save
a program it had to compile, and compiling one blocks the chat.

What a coding client sends. opencode 2.2.5's main request (10,678 tokens under LFM2's template with its 12 tools) has `stream`,
`stream_options` and `max_completion_tokens` 32768; its title request, 10 ms later (471 tokens), `stream`, `stream_options` and
`reasoning_effort`, which the server ignores. Neither has a temperature, `top_p`, `top_k` or penalty, so both take the server's
default sampler, warmed. A POST's `model` is not looked at (a config naming `qwen36-35b` works); `GET /v1/models` answers
`lfm2-2.6b` with `max_model_len` 128000. Past 128,000 tokens the server answers 400, so a client that compacts by its
`limit.context` needs that number: 128000, not the 35B's 262144.

A sampler outside the warm (key (kmax, pen): another top-k bracket, or no penalty) compiles on the chat's thread before its
first token: its program and four wait entries a bucket, 4.3 s at 13,000 tokens and 13.1 s at 100,000 on the build host (the card host
about 2.8x), for every bucket of a chain ring (`Engine._generate_chain`), 14 with a `max_tokens` that long: 64 s and 187 s.
Once a process: a disk under the floor keeps none of it. A temperature or `top_p` of its own stays in the warmed key.

One slot, a request beside another: a new request waits for the running prompt's prefill, then takes the device; the running
one's context goes to the store and comes back from it, not fed again (`test_a_title_beside_the_main_prompt_feeds_nothing_again`).
`lfm_check.py` is the window's client: replies against a server's over the ISA simulator, decode and prefill, captured
bodies replayed.

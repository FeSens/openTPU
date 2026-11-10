# otpu-serve: an OpenAI-style server on the card

`otpu-serve` serves one model on the card to any number of clients over an OpenAI-compatible
HTTP API. It keeps every user's context, on the card while it runs and in the host's RAM and
on disk after that, and it decodes several users' requests together in one run of the card.
In that it follows SGLang: a prefix cache of contexts (SGLang's RadixAttention), a hierarchy of
memory under it (its HiCache: GPU, host RAM, disk), and continuous batching.

```sh
otpu-serve --model qwen3 --backend board                     # http://127.0.0.1:8000/v1
otpu-serve --model qwen3 --backend board --kv-ram-gb 8 --kv-disk-gb 32 --kv-dir /data/kv
otpu-serve --model lfm2 --backend board --port 8001          # one sequence: requests in turns
```

Any OpenAI client works against it (the `openai` Python package, checked: models, chat plain and
streamed with usage, completions):

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="none")
r = c.chat.completions.create(model="Qwen3-0.6B", messages=[{"role": "user", "content": "Hi!"}])
```

## The API

| Route | |
|---|---|
| `POST /v1/chat/completions` | the chat template as otpu-chat renders it; `stream` (SSE), `n`, `max_tokens` / `max_completion_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop` (strings), `stop_token_ids`, `ignore_eos`, `stream_options.include_usage`, `chat_template_kwargs.enable_thinking`, `skip_special_tokens`, `tools` and `tool_choice` (below); content as a string or text parts (and `image_url` parts and `max_slices` on a model with a vision side) |
| `POST /v1/completions` | a prompt string or token ids, `echo`, the same sampling and stops |
| `POST /generate` | SGLang's native call: `text` or `input_ids` (one, or a list), `sampling_params` (`max_new_tokens` (default 128), `temperature`, `top_p`, `top_k` (-1: all), `repetition_penalty`, `stop`, `stop_token_ids`, `ignore_eos`, `n`), `stream` (each event the whole text so far); `output_ids` and `meta_info` (prompt, completion and cached tokens, `finish_reason`, `e2e_latency`) |
| `GET /v1/models` | the model, its context length (`max_model_len`) |
| `GET /v1/stats` | the queue, throughput, the slots, the KV store and prefix cache, the steps' timing |
| `GET /health`, `/health_generate` | ok; a streamed MoE's 503 `warming` while its expert pool is read into the page cache (below) |
| `GET /get_model_info`, `/get_server_info` | |
| `POST /flush_cache` | forget every saved context and what idle slots hold |
| `POST /api/fan_gains` | one fan's kp, ki, kd and optionally die target for fan_pid (`{"fan": 1, "kp": 4, "ki": 0.1, "kd": 0, "target_c": 60}`, JSON content type; docs/dashboard.md) |
| `POST /tokenize`, `/detokenize` | vLLM's: a `prompt`'s or the `messages`' tokens as the completion routes render them (`add_generation_prompt`, default true; `chat_template_kwargs`), `count`, `max_model_len`; `tokens` back to text. Nothing runs on the device |

`usage.prompt_tokens_details.cached_tokens` counts the prompt tokens the server did not feed
(the card or the store held them). Sampling defaults are the model's (otpu-chat's: the
checkpoint's generation_config.json, or its family's), and a request's fields override them. A
top-k above 64 (or none) is 64, the most the card's sampler takes. `--reasoning-parser qwen3`
moves a reply's `<think> ... </think>` to `reasoning_content`, also while it streams. `--api-key`
(or `OTPU_API_KEY`) requires `Authorization: Bearer <key>`. Not supported: logprobs,
`suffix`, images on a model without a vision side; `presence_penalty` and `frequency_penalty`
are ignored.

### A request's own sampler

The generate loop's programs are compiled per context bucket and per sampler: greedy, or top-k's
band of 8 (1-8, 9-16, ... 57-64) with or without a repetition penalty. Temperature, top-p, top-k
inside its band and the penalty's value are the programs' arguments: they compile nothing. A warm
covers the server's default sampler and a request's `temperature: 0` (`warm35x2.dense_samplers`);
the startup's line `samplers on disk: ...` and `/api/stats`' `server.warm.samplers` name them. A request
with another band compiles its programs and computes their wait entries on its first use in the
process, for up to 16 buckets ahead as its `max_tokens` reaches, and its first token waits for that;
the same sampler after it does not (saves off: until the process ends; saves on: they stay on disk).

Measured on the two-card LFM2-2.6B server (2026-11-05, 10.7k-token prompt, bucket 42, saves off;
default sampler, warmed: first token 0.58 s):

| request | first token | compiled |
|---|---|---|
| `top_k` 20, `max_tokens` 60 | 10.91 s | 1 bucket, 4 entries |
| `top_k` 12, `max_tokens` 32768 | 169.37 s | 15 buckets, 60 entries |
| the same again | 0.40 s | none |

A bucket compiles slower further along the context (on a build host 4.3 s at 13k tokens, 13.1 s at
100k). A client sending its own `top_k` either stays in a warmed band or
pays this once per server process.

### Tool calls

OpenAI tool calling (`opentpu/serve/tools.py`) on the format the chat template writes (an agent
such as opencode drives everything through it): the qwen3_coder format of Qwen3.5's and
Qwen3.6's templates, the one vLLM's `qwen3_coder` parser reads, as below, or LFM2's (where the
template holds `<|tool_call_start|>`; "LFM2's calls" after this list):

- **The request.** `tools` (OpenAI's `{type: function, function: {name, description,
  parameters}}`) go to the chat template as given: its `<tools>` block is each one's
  `tojson`, as vLLM and transformers render them. `tool_choice` `"none"` renders no tools and
  parses nothing; `"auto"`, `"required"` and a named function are all auto: the model chooses,
  nothing forces a call. `parallel_tool_calls` is ignored (every call is returned). An
  assistant message takes `tool_calls` with `content` null or `""` (each
  `function.arguments` a JSON object's string, or the object); a `tool` message's content is a
  string or text parts (the template puts consecutive results in one user turn).
- **The reply.** `<tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>
  ...</function>\n</tool_call>`, once or more: `message.tool_calls`
  (`[{id: "call_<hex>", type: "function", function: {name, arguments: a JSON string}}]`), the
  text before the first call `content` (null without), `finish_reason` `"tool_calls"`. A
  value takes its parameter's JSON-schema type: a string as is (the format's newline either
  side dropped, those inside kept), integer, number, boolean, object and array by
  `json.loads`, the text when it does not parse (or the schema does not name the parameter).
  A `<tool_call>` without a `<function=...>` stays content; a reply cut inside a call returns
  what it has.
- **Streamed.** Content deltas as they come, less a possible partial `<tool_call>` and the
  whitespace before a call; a call is held until its `</tool_call>`, then one delta
  `{tool_calls: [{index, id, type, function: {name, arguments}}]}` carries it whole; the last
  chunk's `finish_reason` is `"tool_calls"`, then the usage (`include_usage`) and `[DONE]`.
- **The next turn reuses the context.** The agent's next request re-renders each call from
  its arguments (the parameters in their order, a string as is, the rest `tojson`): the text
  the model wrote, so the prompt continues the context the card or the store holds, and only
  the tool results are fed. The exception is a non-string value written otherwise than
  `json.dumps` writes it (`{"a":1}` for `{"a": 1}`): the prompt parts from the context there.
  On a laptop (a scripted worker, Qwen3.6's tokenizer), opencode 2.2.5's first request
  rendered 10,996 tokens (13 tools' schemas, then its system prompt); after the read tool ran,
  its next request's 11,112 tokens began with all 11,053 of the first's prompt, reply and
  `<|im_end|>`.
- **The thinking kept** (`--preserve-thinking`, otpu-serve35's default; `--no-preserve-thinking`
  turns it off; a request's `chat_template_kwargs` `preserve_thinking` sets it for that
  request). By default Qwen3.6's template writes an assistant turn's think block only for the
  turns after the last user query. So the user's next message re-renders every turn since the
  previous one without its block: with thinking off, without the empty `<think>\n\n</think>\n\n`
  the generation prompt opened. The prompt then parts from the context at the first of those
  turns, and every tool step since the previous message is fed again. With the template's
  `preserve_thinking` every assistant turn keeps its block, as the model wrote it (its
  `reasoning_content`, when the client sends it back, goes to the template). Each request's
  prompt is then the last prompt, its reply and `<|im_end|>`, followed only by what is new: the
  tool's result, or the user's next message. Only a template with the switch gets it (the
  server checks the template's text). `test_serve_tools`: an opencode-shaped session of three
  requests, in characters and in Qwen3.6's tokens.

LFM2's calls are one block of Python-like calls, `<|tool_call_start|>[read(filePath='a.py',
limit=20), bash(command='ls')]<|tool_call_end|>`, read with Python's parser and its literals alone
(nothing evaluated; `true`, `false` and `null` too), positional values by the schema's order. The
model also writes what Python does not read: the checkpoint's own reply to a request for two
calls is `[bash(command=ls -la /tmp), read(filePath=/work/a.txt)]`. So a block that does not parse
is read loosely for the request's tools only, each value up to the next argument or the call's
end, by its parameter's type; the JSON form `{"name", "arguments"}` (or a list of them) is taken
too. A block that is none of these, or is cut short, stays content. The template renders a call
back with single quotes, so a call the model wrote that way continues the context exactly, and
one it wrote otherwise parts from it at the call: its tokens are fed again. The whitespace before
a block stays in `content`, since this template writes none of its own there.

**Which models call tools well.** The parser is not the limit; the model is. A screen on
opencode 2.2.5's own requests (2026-11-06): 8 prompts in a one-file project (what a file prints,
read it, list the files, add a line, run it, find the Python files, search for TODO, create a
file), on opencode's default config (12 tools, a 10.5k-token prompt) and a small one (10 tools,
2.9k), the first call and the turn after a tool that opencode really ran. Each model's own chat
template, thinking off, greedy, fp32 from the reference implementation. *Valid*: every call names
a listed tool with its required arguments. *Right*: the first call does what was asked. *Next
turn*: the answer reports the tool's result, with no invented file or output.

| model | config | valid of 8 | right of 8 | next turn right of 8 | |
|---|---|---|---|---|---|
| Qwen3.6-35B-A3B | default | 8 | 8 | 8 | |
| | small | 8 | 7 | 8 | |
| Qwen3.5-4B | default | 8 | 6 | 7 | Run goes to `execute` instead of `shell` |
| | small | 8 | 8 | 7 | |
| Qwen3.5-2B | default | 7 | 6 | 7 | |
| | small | 8 | 7 | 7 | |
| gemma-4-E4B-it | default | 8 | 7 | 4 | after `glob` and `write` its reply is empty |
| | small | 7 | 7 | 5 | |
| LFM2.5-8B-A1B | default | 8 | 4 | 4 | repeats the call it just ran |
| | small | 7 | 6 | 4 | one made-up output |
| LFM2-2.6B | default | 3 | 2 | 4 | tools and arguments not in the list; `edit` on requests to read |
| | small | 7 | 2 | 4 | invents files and results |
| SmolLM3-3B | default / small | 2 / 4 | 2 / 4 | | mostly no call |
| Phi-4-mini | default / small | 1 / 7 | 1 / 7 | | most default calls invalid; the small config's 7 counts bare and fenced JSON, which the server does not parse |

By valid >= 7, right >= 6 and next turn >= 6 on both configs with no unrequested edit or write (a
rule set after seeing the numbers), the 35B, Qwen3.5-4B and Qwen3.5-2B pass; all three write the
qwen3_coder format. Eight prompts and one path each: 1 of 8 means nothing, and those three are not
ranked by this.

- **LFM2-2.6B is not usable with opencode, and it is neither the server nor the format.** The
  prompt is rendered token for token as the reference renders it. On the device's formats
  (emulated: 8 prompts as sent, 40 sampled paths each) valid calls are 99 of 320 on fp32, 78 on
  the card's mix and 80 on int8. It takes argument names from habit (`file_path` for `path`) and
  tool names from the system prompt's prose. A smaller system prompt fixes the names (268-271 of
  320 valid), but the model still chooses the wrong tool, and on the turn after a tool it
  ignores the output for a likelier answer and reports edits it did not make (right on 147 of
  320 paths).
- **Qwen3.5-2B on the card's formats** (the device's quantization emulated at four slices; float
  reproduces the reference's 32 replies word for word). Greedy valid / right / next turn, default
  then small: fp32 7 / 6 / 7 and 8 / 7 / 7; int8 6 / 5 / 7 and 8 / 7 / 6; 4-bit 6 / 6 / 7 and
  8 / 6 / 6. Neither format drops 2 in a cell or takes an unrequested action in its 1,280
  sampled paths. What the formats lose is answers that need the model to evaluate code: on the next
  turn, sampled, "What does hello.py print" is right on 27 / 0 / 1 of 40 paths for fp32 / int8 /
  4-bit. int8 is the one to serve: 4-bit's +51 % speed costs it 7.79 points of KL divergence
  (docs/formats.md).
- **The 35B on the live server** (two cards, fp4 weights with an int8 head): 8 / 8 / 8 on the
  default config and 8 / 8 / 8 on the small, against fp32's 8 / 8 / 8 and 8 / 7 / 8; 71 sampled
  paths on the small config (the server's default sampler) all valid and right; no unrequested
  action; `prompt_tokens` equal to the reference's count on all 103 requests. Four or five paths
  a body cannot see a rare failure. First token from zero 162.6 s at 10.5k tokens (prefill 64.9
  tok/s, decode 57.2) and 44.5 s at 2.9k (66.1, 62.1); a real next turn 1.31 s with 2,974 of
  3,027 tokens cached.

Limits of the screen: fp32 and greedy except where said, one tiny project, one opencode version,
one step of the next turn.

### Images (MiniCPM-V)

A model with a vision side (`minicpmv.Spec`, docs/minicpmv.md) takes images in
`/v1/chat/completions`:

- **Parts.** A message's content may hold `image_url` parts (`{"type": "image_url", "image_url":
  {"url": ...}}`, or the Responses API's `input_image`) between its text parts, in any order.
  The chat template renders each as `<|image_pad|>`, and the image's placeholder text (its
  `<image_id>`, thumbnail and slices) takes its place, as otpu-chat and HF's processor do.
- **URLs.** `data:` URLs (base64 PNG, JPEG, WebP) by default. `--image-fetch` also fetches
  http(s) URLs (20 s, 32 MB). A server never opens one of its own files for a request.
- **Slices.** A request's `max_slices` (1 .. 36) bounds an image's slices besides its thumbnail;
  `--max-slices` is the server's default, else the processor's 9. It is the latency knob: each
  crop is about 3.7 s of the card's vision tower at MCOLS 4 and 64 prompt tokens.
- **Keys.** Every image's `<|image_pad|>` is one token id, so a context's key holds each image
  token's stand-in instead (`minicpmv.keyed`: a negative int32, a hash of the image's content
  key, the sha256 of its crops' sizes and patch rows, and the token's index). The device's
  context, the prefix cache and the store compare keys: a request reuses a saved context past
  an image only for the same image at the same slices. The worker feeds the token ids, picks
  with them (the repetition penalty), and keeps the keys in the context.
- **Feeding.** The images with tokens in the part to feed run through the card's tower first
  (an image that started before the reused prefix, whole); then the prompt is prefilled with
  its image rows (`Engine.prefill_layers(vis=...)`).
- **RoPE rows.** At each switch the worker writes the request's M-RoPE rows into the RoPE table
  (only those that differ; the table's own rows back for a text-only request). The model
  decodes one sequence at a time (no batched slots), so the table is the running request's.

`/v1/completions` and `/generate` take text and token ids only. An image request on a model
without a vision side is refused (400).

**A streamed MoE's expert pool** (`--pool`) serves the slots' misses from the host's page cache:
~0.5 ms a miss, against 15-32 ms each from the card host's disks (docs/minicpmv.md 6, pooldisk:
a turn 5-9x slower). The engine reads the whole pool into the page cache as it opens, and again,
once the image is built, what the build's memory took back out (`moe.rewarm`): the card host's
HDD reads an 18 GB pool in ~80 s, its SSD in ~37. Meanwhile `/health` answers 503
`{"status": "warming", "progress": <the share read>}`, for at most `--warm-cap` seconds
(default 300; 0: never); past that, or when the read failed or was stopped, it answers ok with
`"warm": "incomplete"` and a note. It is advisory: requests are served during the read, their
misses from the disk. The read's start, end, GB, MB/s and the pool's share in the page cache
at its end are printed when it ends, and `/v1/stats` has them (`pool_warm`).

## The dashboard and /metrics

`/` is a live dashboard (throughput, TTFT per request, the prompt in flight with its ETA, the
queue, the KV context and store, the card's utilization, temperature and power, the host), and
`/metrics` Prometheus' text format with vLLM's names (`vllm:` prefix) and `otpu:` for the card,
the store and the host; `/api/stats` and `/api/events` are the page's data. `--telemetry 2` reads
the card (otpu-smi's default reads, its own thread); `--demo` serves the page over simulated
data. docs/dashboard.md.

## The KV cache: pages, the store, the prefix cache

**A context's bytes** (`opentpu/serve/kvio.py`). Every positional KV cache keeps the positions
`[0, n)` in contiguous byte ranges:

- K's first n rows, and its scales;
- V^T's first `ceil(n / 256)` tiles, since V^T is stored in tiles of 256 tokens, each `[d, 256]`
  contiguous (`compiler.KVDesc`);
- V's first n scales.

A **page** is 256 positions of every cache of every layer: 15.1 MB for Qwen3-0.6B. Page i
holds what the tokens `[0, 256 (i + 1))` put there and nothing else, so contexts with a common
prefix share its whole pages. A context of n tokens is:

- its whole pages;
- its last partial page;
- the state the KV caches do not hold (`Image.state_regions`: Qwen3.5's DeltaNet states, LFM2's
  convolution rings, Gemma 4's sliding rings).

Writing the pages back and setting the position (`Engine.adopt`) gives the context bit for bit,
as `Engine.rollback` does. On the card, a save or restore is staged: LD / ST copy programs
gather a page's ~900 regions into an area of the DRAM past the program area, and one DMA moves
it. Each region's own DMA would cost ~30 us of set-up. The state is staged too, in batches of
the area's room. A save at a snapshot's position (the idle save, `_save_at`) reads the state
from the snapshot instead (`kvio.snapshot_at`: the same bytes in one range, no copy).

What that buys on the 35B (bcu1525-wg:bhw16, one slice of 8 channels x 128 B). A host profile
(a real `Board` over a transport that counts the XDMA calls; the 35B's layout built from its
Spec without weights) of a 100-token context:

| | XDMA calls | MB | copy runs | host s (a laptop) |
|---|---|---|---|---|
| tail page, read or write (80 regions) | 8 / 11 | 1.87 | 1 (161 instructions) | 0.001 |
| state, direct (480 regions: 450 windows of 6 KB, 30 of 1.05 MB) | 3,840 | 34.4 | 0 | 0.02 |
| state, staged | 24 read / 8 write | 34.4 | 1 (1,921 instructions) | 0.02 / 0.005 |
| state from the snapshot | 24 | 34.4 | 0 | 0.02 |

The card before (2026-11-03): a restore 0.153 s, an idle save 0.175-0.188 s. Those are ~3,850
XDMA calls at ~30 us each (0.115 s), plus 34.4 MB over the ~2.9 GB/s link (12 ms; 20 ms
through the bounce buffer for an unaligned write), the host's ~0.02 s and, for the save, the
rollback's copy (~0.021 s, the card's `snapshot` phase). Predicted after: a restore ~0.04 s (the
link's 12 ms, the scatter copy ~0.021 s, the host), an idle save ~0.06 s (the rollback ~0.021 s,
the state's 34.4 MB from the snapshot 12 ms, `hash_swap` and the copies on the host ~0.02 s). The
copy runs are the guess: ~0.021 s for 34.4 MB is what the card's snapshot shows (its
`snap_stats` cycles say whether that is the device or the run's set-up). An area smaller than
34.4 MB takes a copy per batch.

**Lanes** (`kvio.Layout.relane`). On a machine with the channel hash (`Config.CHASH`) K's and V^T's
group chunks hold their lanes permuted by their DRAM address (`quant.hash_rows`), so the same context
is other bytes in each sequence's caches of an `Image(batch=N)`: on the two-card 35B's image only
64-128 of page 0's 1,280 chunks stand in the same lanes in slots 1-3 as in slot 0. The store holds
pages in the first sequence's lanes, and `read_pages` and `write_pages` move them for the other
slots, so any slot restores what any other saved. Before this a context saved in one slot and
restored in another came back with K and V^T scrambled and no error (the logits differ; same-slot
restores were right). `copy_from` goes through the host there: a copy on the card cannot move lanes.
The store's directory is keyed by it (`"lanes"` in the model key, `api.store_key`), so entries saved
before are not read; a single sequence's key stays.

It reaches an image with slots on a target with groups and the hash (`bcu1525-x2`, `bcu1525-s2`,
and `bcu1525-wg` under `--slots` above 1). `ypcb-00338`, `bcu1525` and `nangate45` have
`WG` 1, so nothing moves, and a single sequence is the first. `relane` runs on the host, on the
worker's thread, inside the read or write, so it stalls the other slots only as part of the save or
restore: on a build host 2.4-3.5 ms for a cut's 2 new pages in a save of 34-42 ms, 50 ms for a 10,509-token
restore's 41 pages, nothing for slot 0.

**The store** (`opentpu/serve/store.py`, `BlobStore`). Pages and tails live in host RAM up to
`--kv-ram-gb`. Past that, the least recently used go to files under `--kv-dir`, written by a
thread, up to `--kv-disk-gb`; past that they are dropped. A blob read from disk comes back into
RAM. An active request's blobs are pinned. The entries are written through to disk, and at
shutdown the RAM goes too, so a server started on the same directory continues the contexts
saved before. The directory is keyed by the model, its formats and layout. An entry restored
twice (`store.BOOST_HITS`: a system prompt that conversations share) is boosted: its blobs,
and the pages it shares, go after every other in their tier and in the entries' limit, so a
long chat cannot push it out (`kv_store`'s `boosted`). otpu-serve35 keeps a disk tier by
default: `--kv-disk-gb 64` under its `--kv-dir`, cut to what leaves 32 GB free on
that filesystem (the store's own files counted as room; under 1 GB none), so the prefixes and
contexts outlive a restart.

At its limits the store drops whole entries: a blob evicted takes every entry that uses it, and
those entries' other blobs with them. Two bugs there were fixed (2026-11-06); the first was
seen on the cards with the store filled to its disk limit, the second by a test, and the fixes
have run at the limit in tests only. The eviction walked a list of keys made up front, and
a key a drop had already taken raised in the writer thread, which died: blobs leaving RAM then
piled up unwritten and new contexts never reached the disk, while restores still worked and
`errors` stayed 0 (by the code, a stop would then wait on the writer's queue for good; not
observed). And with the writer alive, its drop (the store's lock, then the prefix cache's) could
deadlock against a save (the other order): a test that forces the timing hangs, the cards never
showed it. Now
both walks pass keys already gone, the prefix cache uses its store's lock, and the writer counts
and logs any exception and goes on: `kv_store.writer_errors` and `kv_store.writer_alive` in
`/v1/stats`. A store left part way loads clean: `.tmp` files are removed, an entry is kept only if
all its blobs are there at their sizes and its record hashes to its key, a directory over its
limit is trimmed after the load (boosted entries last) instead of before any entry is known, and
a restore that finds a blob gone or short writes nothing to the device and the request feeds from
the start. `tests/test_serve_store.py` holds each case; `tests/test_serve_prefix35.py` runs a
store left part way through the worker.

A load keeps the entries whose blobs are all there, each at its size. A record short of one, blobs
no entry uses and a writer's leftover `.tmp` files go, then the disk comes back within its limit,
so that boosted blobs go last and a dropped blob takes its entries with it. A restore has every
blob in hand before its first write to the card: a blob gone means a feed from the start, never
part of a context. The store and its cache share one lock, because the writer thread drops entries
and a save puts blobs. `kv_store`'s `writer_errors` counts what that thread failed at (the traceback
goes to stderr) and `writer_alive` says whether it runs; neither is in a request's `errors`.

**The prefix cache** (`PrefixCache`). Entries are the contexts saved, by their tokens; a page's
key hashes the tokens up to its end. `match(prompt)` finds where the card can continue a
prompt:

- for a model whose KV caches hold the whole context, any common prefix of a saved context (its
  pages up to there; what lies past is overwritten);
- for a model with state, the end of a saved context that the prompt continues. So the
  scheduler keeps prefixes for that too (below: "Prefixes for other conversations").

A prompt always feeds its last token, whose logits pick the first one of the reply.

## Requests at once

**Batched, in KV slots** (`opentpu/serve/batching.py`, `BatchWorker`: Qwen3 and the Llama-likes,
`--slots`, default 8). The image holds B sequences' KV caches (`Engine(batch=B,
batch_area=True)`). Each iteration of the worker:

1. Waiting requests take free slots: one in the block of MCOLS slots holding the most
   requests (a step runs each block in a run of its own, so four requests in one block take
   one run a step, spread over two they take two); then the one holding the longest prefix of
   the prompt, slot 0, the least recently used. The prompt continues the longest prefix found
   in that slot, in another slot (its pages copied on the card, no PCIe) or in the store. A slot's
   context is saved to the store before it is overwritten.
2. Feed runs of the prompts being fed, oldest first, up to `--chunked-prefill-size` tokens
   (default 64, SGLang's chunked prefill; more: a shorter time to first token, longer pauses
   of the decoding requests). A feed run takes up to MCOLS (4) tokens of a prompt, every
   weight streamed once, no LM head (`Steps.feed`). A prompt feeds all but its last token
   this way.
3. One batched step for every decoding request (`opentpu/llm/batch.py`):
   - a run feeds each one its token from DRAM; the embedding rows are gathered and the RoPE
     rows loaded from the image's tables;
   - every weight streams once for all of them;
   - the LM head hands each chunk to every row's sink: the decode loop's `Greedy`, or its
     `Sampler` with the slot's words (top-k, temperature, top-p, repetition penalty) and the
     request's uniform;
   - the picks go to DRAM.

   A request's first step feeds its prompt's last token, so its first token is picked on the
   card too. The host writes the tokens and reads one word per request: no logits leave the
   card, and the host computes nothing of the model.

A request decoding alone in slot 0 runs the card's decode loop instead (`generate_card`),
halted after the token in flight when another arrives. A step's rows run in groups whose program
fits IMEM. The LM head's sinks and each row's attention grow it, so four rows of Qwen3-0.6B fit
early in the context and fewer late in it.

**Span programs.** A step's program would take its rows' positions at compile time, and the
host's compile of a 4-row step (~50-130 ms on the card host's i7-4790, growing with the context)
took longer than the card's run of it (44-59 ms). Instead, a program serves its rows for a span
of steps: row r is at position p_r + t, where t is the run's step in a DRAM word that the
program loads and multiplies into its argument and address registers (RLD, as the prompt runs
do). All rows add the same t, so they share those registers. The span ends where a row reaches
the end of its attention block (256 tokens); until then the same program runs again, already
in IMEM. The next one is compiled ahead as soon as a program starts serving, and so are the
steps whose rows the worker can tell ahead (a prompt the feed budget finishes, a request
reaching max_tokens). A step's rows run in blocks of MCOLS slots (an MXU pass takes MCOLS rows:
more rows in a run cost another pass), so a request that leaves changes its block's program
only; and until that block's new program is compiled (for a few steps on), the live one runs
on with the free slot's row fed a dummy token past its saved context, its pick dropped. A feed run's
program serves its slot's runs through an attention block the same way, its rows advancing by
MCOLS a run.

**Mixed runs** (`--mixed`, off by default). With fewer decoding requests than MCOLS, a step's
MXU pass has spare rows, and a prompt being fed can take them: one run of the decoding rows
and the oldest prompt's next tokens (`Steps.mixed`). Its program takes two run-time values
(the decoding rows advance one a run, the feed rows as many as they are) and has its feed
rows at their attention block's start, so it serves the prompt's runs. While it compiles, the
iteration does one feed run. The decoding requests then never wait for a whole feed budget;
but a prompt advances only by the spare rows a run, so it takes longer to its first token.

**Batched, in sequence slots** (`opentpu/serve/batching35.py`, `BatchWorker35`: a Qwen3.5-MoE
whose image holds N sequences, `--slots N`; docs/batch35.md). Each slot is a sequence's KV caches,
DeltaNet states and windows, at 262,144 / N tokens unless `--cap` says otherwise. A request takes
the slot whose context it continues, else the store's longest match restored into a free slot,
else position 0. Prompts go in by layer runs at the slot, up to 512 tokens an iteration, between
the decoding requests' steps; one step advances every decoding slot, each token picked on the
card; a request decoding alone in slot 0 runs the card's decode loop. A step's program is keyed by
its slots and sampling kinds only, each row's position a run-time word, and every one is compiled
before the server listens (`warm_steps`), so a request joining or leaving waits for no compile.
Each request's tokens are a single-sequence engine's, greedy or seeded. On two BCU1525s with four
slots (2026-11-05): one request decodes at 65 tok/s; with four the device steps at about 103
tok/s in all (its rate while stepping at 3.88 rows a step on one 64-token leg; by the wall clock
four 256-token requests summed 81.3 tok/s of decode); four chats arriving together are bound by their prompts, which are fed one
budget part at a time. A step is DRAM-bound and its rows' own work does not overlap the shared
stream, so the total grows well under 4x (docs/batch35.md sections 4-7).

**In turns** (`opentpu/serve/scheduler.py`, `Worker`: LFM2, Qwen3.5, Gemma 4, the streamed
MoEs, whose images hold one sequence). The requests (`--max-running` at once) take the card in
turns of `--quantum` tokens. Each switch saves the outgoing context's new pages and tail and
restores the incoming one. A request running alone runs to its end. The prompt is fed but its
last token, which the card's decode loop feeds and picks from (when the loop takes the request's
sampling; else the host picks the first token from the logits).

A prompt is kept where the next turn's rendering parts from it (the reply's start, `at`), and
only there: a model with state (Qwen3.5's DeltaNet) can continue a context only at a position
it kept. With engine snapshots (`Engine(snapshots=)`, the 35B's) that keep is a snapshot on the
card (a DRAM-to-DRAM copy of the state and the convolution windows, no PCIe), not a save to the
store; the reply decodes past it. Then:

- the same chat's next turn (its prompt goes on from `at` with other tokens) rolls the device
  back to the snapshot (`Engine.rollback`) and feeds its new tokens: no store traffic, as
  otpu-chat does between turns;
- a request that parts from the device's context elsewhere, or a switch to another request,
  first rolls back to the snapshot and saves that context to the store (the save chat35 does
  not have to make);
- when the worker goes idle, it does that save then (`stats["idle_saves"]`), off every
  request's path: the next request finds the context on the device or in the store either way.
  A request that continues the context being saved (the chat's next turn, sent the moment the
  reply ends) pre-empts the save: it gives up between two device reads (`kvio.ContextIO.abort`;
  nothing stored, the device as it was), the request goes on from the device, and the next idle
  saves what the device holds then (`stats["idle_preempts"]`). Any other request waits for the
  save, which its own switch would have to make: 0.18 s for a 34-token context on the card
  (serve35_check, 2026-11-03: r2's and r3's queue 0.184 / 0.182 s, r3's restore of the same
  size 0.155 s; the state's 3,840 XDMA calls direct, ~0.06 s predicted staged and from the
  snapshot, above), its record's `idle_save` phase.

Without snapshots (the engine has none, or its area is full) the keep is a save to the store,
as before. Each request's record has its time to first token's parts (`phases`: `switch`
and within it `save`, `restore`, `rollback`; `snapshot` within the prompt; the queue time is
the record's `queue_s`, `idle_save` the part of it an idle save held up), in `/api/stats`' recent requests and the
dashboard's TTFT tooltip. `/flush_cache` forgets the store and, when idle, the device's context
(a flush during an idle save forgets that save too).

**A reply that calls a tool stays on the card.** An agent's next request goes on from its call.
Qwen3.6's template renders an assistant message after the last user query with its think block
(with thinking off, the empty one the generation prompt opened), then the call as the model wrote
it ("Tool calls" above), then the tool's result. So with tools the API sets the request's
`keep_reply` (`"<tool_call>"`), and a reply whose text holds a call is not rolled back when the
worker goes idle. The idle save writes the reply start from the snapshot (`Worker._save_reply`,
no rollback), then the whole context, and leaves the card at its end. The agent's next request
continues it from the card and feeds only the tool's result. A request that parts at the reply
start restores that entry; with the card holding its pages, only the state is written.
Without `--preserve-thinking` the user's next message parts there, because the template drops
the empty think blocks of the turns before it. With it (otpu-serve35's default) the next turn's
rendering parts nowhere, so the API keeps every reply (`keep_reply` `""`) and gives the reply
start at the prompt's last position (its last token fed after the snapshot): a reply that
tokenizes otherwise in the next prompt, or that the template trims, costs that token and the
reply, not the turn. A switch to another conversation before the idle keeps a kept reply the
same way (the reply start, then the whole context, to the store). Before this, the first
request after each user message fed the reply again: tens to hundreds of tokens of the call, at
the 35B's ~60 prompt tokens a second.

**Prefixes for other conversations** (`Worker.prefix_block`, `--prefix-block`, default 2048;
`Request.marks`). A model with state continues only at a position it kept. Without more, a new
conversation that shares another's system prompt would feed all of it again: opencode's tools
and instructions are 10,976 of its first request's 10,996 tokens, ~180 s on the 35B. So the
worker splits the prompt's feed and keeps the context at each cut as a store entry with no reply
(`Entry.prefix`, its blobs on disk at once). The cuts:

- the mark the API gives (`api.prefix_marks`): the `<|im_start|>` of the final user turn, so
  the context holds every message before it (an agent's first request: its tools and system
  prompt). The API finds it by rendering the messages before that turn with a stand-in user
  message, then taking the last special token (the tokenizer's `added_tokens_decoder`) before
  the two renderings part;
- the multiples of the block. A prompt whose system prompt's tail differs (today's date)
  restores the last block before the change;
- none within `PREFIX_SKIP` (256) positions after the feed's start, nor at the prompt's last fed
  position. The reply start `at` is kept as before.

A new conversation's prompt matches the longest of them. Each cut is a save mid-prompt: the
`prefix_save` phase, `stats["prefix_saves"]` and `prefix_save_s`. Restores of these entries are
the prefix cache's `prefix_hits` and `prefix_hit_tokens`. They also show as
`otpu:prefix_entry_hits_total` and `otpu:prefix_entry_hit_tokens_total`, and as the dashboard's
"shared prefixes restored".

The split costs each part's set-up: draining the pipeline, and the expert server's
`end_prefill` and `begin_prefill`. The next part's first MoE layer streams its experts without
the previous layer's run to hide behind (~0.2-0.3 s). With the feed starting at 0, the blocks
fall on the layer-major chunks' boundaries (512 rows), so they add no chunk. The mark adds one
short chunk, and its last few tokens run as decode steps: 16 for opencode, ~0.5 s. Each save
moves the state (34.4 MB) and the pages the store lacks. On the two-card 35B at block 512,
opencode's first request took 6.27 s more on 162.53 s, +3.9 % ("On the cards" below).

Not done: keeping the state as the layer-major pass crosses a block, with no split. The chunks
of `img.prefill_rows` each pass every layer before the next. So between two chunks the card
holds the state at the first chunk's end, and a callback there (`prefill_layers` to the worker)
could read the state. The pages could wait until the prompt ends, since later chunks do not
write them. The mark would still end a chunk early. This change is in `opentpu/llm/qwen3.py`,
which is part of the card image's key.

**In sequence slots** (`BatchWorker35`, `opentpu/serve/batching35.py`). The same cuts, taken by the
feed: a run stops at a prompt's next cut, the slot's context goes into the store through that slot's
`ContextIO`, and the iteration goes on to the decoding requests' step. With the feed starting at a
multiple of 512, block 512 adds no chunk (`PREFILL_BUDGET`); the mark adds one short chunk. Any
position stays on the warmed programs: a run's key is its bucket and its rows (`warm35x2.layer_keys`,
`test_new_sessions_restore_the_shared_prefix`). A prompt's reply start (`Request.at`) is one more cut,
kept without `prefix`, since a slot has no snapshots: the conversation's next turn restores it,
whatever the reply renders as. A prompt that continues a free slot's whole context takes that slot. Any other takes slot 0 if it is
free (the card's loop runs there), else the free slot sharing the most tokens with the prompt, the
least recently used of equals. Tokens, not whole pages: chat prompts share their first token, so a
slot holding anything beats an empty one. A restore does not write again the whole pages among the
tokens shared.

The cost at x2 (S = 4; page 5.37 MB, state 34.4 MB; opencode's 10,530-token prompt, ~162 s at
65 tok/s). First the host's part alone, on a build host with the 35B's real image over two cards
that answer at once (the card's and the link's parts are not in it), per prompt:

| `--prefix-block` | cuts | to the store (per card) | host |
|---|---|---|---|
| 2048 | 6 | 429 MB (215 MB) | 0.35 s |
| 512 | 21 | 945 MB (473 MB) | 0.76 s |

| restore of 10,509 tokens | over the link (per card) | host |
|---|---|---|
| into slot 0, empty | 257 MB (129 MB) | 0.10-0.13 s |
| into another slot, empty (`relane`'s part 0.05 s) | 257 MB (129 MB) | 0.24 s |
| into a slot sharing 40 pages | 43 MB (21 MB) | 0.02 s |

What is left of the prompt after a restore runs on warmed programs, also with the program cache's
saves off (free disk under `FREE_FLOOR`): on the warm's cache, in each of the four
slots, nothing compiled and no wait entry computed. The host's part, reading programs
and entries from disk, then again with them in memory:

| what is fed | programs | host |
|---|---|---|
| 10,509 + 21 tokens (bucket 42) | 81 | 0.34-0.36 s, then 0.23 s |
| 10,509 + 16 tokens | 41 | 0.155 s |
| a chunk cut short at a mark, 10,240 + 269 | 122 | 0.60 s |
| a whole chunk, 10,240 + 512 | | 0.59 s |
| 2,900 + 21 tokens | | 0.34 s |
| 426 + 40 tokens | | 0.215 s |

A step's programs do not depend on the position and are live from `warm_steps`.

**On the cards** (2026-11-06: two BCU1525s, the 35B in fp4 with an int8 head, four slots of 131,072
tokens, `--prefix-block 512`, an empty store; greedy; opencode's real request bodies, 10,531 tokens
on its default config and 2,922 on a small one; one run, 41 requests, no errors):

| | host alone (above) | on the cards |
|---|---|---|
| saves, block 512 | 21 cuts, 945 MB, 0.76 s | 21 cuts, 946 MB on disk, 2.40 s in all, 0.114 s each |
| the first request | | +6.27 s on 162.53 s (+3.9 %): the saves' 2.40 s, the feed in parts up to 3.90 s |
| restore into another empty slot | 257 MB, 0.24 s | 257.5 MB, 0.70-0.71 s (host 0.24-0.30, link 0.21-0.24, load 0.15-0.17, run 0.03) |
| restore into slot 0 sharing 3 pages | | 241-242 MB, 0.46-0.51 s (host 0.04) |
| restore with the pages in place | 43 MB, 0.02 s | 37.2-37.5 MB, 0.08-0.09 s |

The +3.9 % has no control on the same server (no `--prefix-block 0` request there; the reference is
the same body's earlier run), and part of it may be the server's first request. One save, the
prompt's last: 42.6 MB in 0.1235 s (link 0.0651, load 0.0358, store 0.0089, host 0.0081, run
0.0051). `prefill_s` leaves the saves out: they show between it and the first token. Moving a
context's lanes for another slot (`relane`) is 0.20-0.26 s of a 10.5k-token restore and 0.13 s of
a 2.9k one.

A new session's first token, against 168.8 s for the same body from zero:

| request | slot | cached / prompt | first token |
|---|---|---|---|
| the same body, a new session | not sampled | 10,530 / 10,531 | 0.27 s |
| the same tools and system prompt, another message (three of them) | 0 | 10,509 / 10,529-10,534 | 1.15-1.71 s |
| the same body while 1, 2, 3 other slots decode | 1, 2, 3 | 10,530 / 10,531 | 0.79-0.80 s |
| the small config's body again, others decoding | 3 | 2,921 / 2,922 | 0.48 s |
| a conversation's next turn (the device's context; a restore) | 0 | 2,974 / 3,027; 10,582 / 10,635 | 1.41 s; 2.07 s |

Every reply with a from-zero reference equals it word for word; the next turns' bodies have
none, and their replies are right by reading. A restore stalls the slots that are decoding
once, by its own length (0.78-0.80 s on the big body); beside another prompt's feed they get about
one token a chunk (gaps of 7.9-8.2 s), the batched loop's behaviour with or without prefixes. A
client dropped mid-prompt freed its slot 8.32 s later with 1,536 tokens fed, and the prompt sent
again fed only the rest.

**Is the restored context right?** An equal short reply is weak evidence: opencode's tool calls
hardly depend on the far context. So a second run planted 51 four-digit facts through the system
prompt and tool descriptions of an 11,130-token body (44 pages; the facts at least 200 tokens
apart) and asked for all of them in order, a reply of 440-449 tokens that needs every page:

| request | slot | cached / prompt | first token | restore | reply |
|---|---|---|---|---|---|
| from zero (two bodies) | 0 | 0 / 11,130 | 176.4-176.8 s | | the references: 48 and 50 of 51 right |
| restored from RAM, others decoding | 3, 2, 1 | 11,129 / 11,130 | 0.78-1.22 s | 0.68-1.12 s | its reference's, word for word |
| restored from RAM | 0 | 11,129 / 11,130 | 0.57 s | 0.50 s | the same |
| restored from disk | 3 | 11,129 / 11,130 | 3.50 s | 3.41 s (the disk's read 2.70 s) | the same |

The wrong digits are the model's own, the same in every restore of a body (the fp32 reference
gets 49 of 51). Three of the 44 pages have no right digit behind them (template text, the
question, and one whose only fact both references miss), so the recitation does not check them. What the recitation would notice was measured apart, on the fp32 model with pages
of its attention K / V cache corrupted: one page swapped for another, its lanes permuted, K alone,
V alone, or a quarter of its lanes, at the context's start, middle and end, each changes the reply
at that page's own fact. One layer of ten swapped on one page does not change it. The ordinary
tool call notices a third of the context scrambled but passes a whole swapped page, 3 of 3. That
control is fp32, not the card's formats, and it does not corrupt the DeltaNet state.

**Across a restart.** After a clean stop (4 s; 180 files, 2.10 GB) a new server loaded 47 entries
from them. A new session of the big body then took its first token in 3.27 s (42 blobs from disk,
2.45 s of reads) against 168.4 s from zero, and the recitation body in 0.86 s into slot 3 with
three slots decoding (its files still in the OS page cache: 0.08 s of reads), both replies their
references'. A request adds about 74.6 MB to the store (two tails; the state alone is 34.4 MB).

A save's and a restore's parts are phases too, `save.<part>` and `restore.<part>`
(`Worker._io_parts`, from `kvio.ContextIO.t`), summing to the phase:

| part | what |
|---|---|
| `store` | the store's own work: the lookup or the insert, its hashes and host copies (a save's settle) |
| `host` | kvio's host work: region lists, slicing, the per-slice concatenation |
| `prog` | building the copy programs (`snapshot.copy_programs`; the state's kept from the first) |
| `link` | the backend's reads and writes: the card's XDMA over PCIe (a bounce buffer when unaligned) |
| `load` | starting a copy: assembling it, its wait entries, IMEM, RUN (`BoardBackend.start`) |
| `run` | the copy on the card until it halts (`BoardBackend.wait`) |
| `adopt` | `Engine.adopt`: precompiles in flight waited for (a restore's) |

Within them, not added: `gc` (Python's cyclic GC, any thread's: `gc.callbacks`) and `telemetry`
(the card's telemetry sample running meanwhile, `telemetry.overlap`: its thread holds the
interpreter for most of a sample, so the worker's host work waits). The record's
`io` has the bytes each moved: `link_bytes`, `link_calls`, `copies`, `copied_bytes` (every
slice's). `/api/stats`' `kv.last_io` keeps the last save's and restore's parts and bytes, an idle
save's included; `serve35_check.py` prints all three.

**With the MTP drafter** (`--mtp`, opt-in: a streamed Qwen3.5-MoE from its -mtp pool,
`--mtp-from` for MiniCPM-V 4.7, or a dense Qwen3.5; docs/mtp.md, docs/offload.md 15.6). Every
request decodes on the card's MTP loop (`MTPDecoder.loop_card`), halted after the token in flight
as plain decode's is. Only sampling the loop takes is served (greedy, top-k 1 .. 64, repetition
penalty >= 1; else 400). What changes against plain decode:

- The whole prompt goes through the decoder's prefill, which appends the MTP layer's K / V rows
  with it. The first token is picked on the host from the last row's logits as the loop would
  pick it (greedy: the argmax; sampled: the device sampler's arithmetic with the request's first
  uniform). Its draft is left to the loop's first run.
- A context's bytes hold the MTP layer's KV cache (kvio's pages: one more attention layer, the
  35B's +1,048 bytes a position, +0.6 / 1.4 / 2.4% of a context of 300 / 1,000 / 2,000 tokens)
  and the DeltaNet states of the decoder's committed slot, restored into slot 0.
- The MTP layer's row at q takes the token at q + 1, which the context does not hold. A page's
  key holds one token more, and an entry the token its last row took (`follow`). A context
  saved for its request to come back to keeps the draft after its next token. A draft still
  pending (a reply that ended right after its prompt) is run before the context is saved.
- The part of a prompt before the reply's start (`at`) is saved with the residual row before it
  (`held`; the streamed MoE's decoder holds it). A prompt that goes on there with another token
  (a chat's next turn) restores that context, and its prefill first appends the MTP layer's row
  at `at` - 1 again from the held row (`MTPDecoder.refill`), bit for bit a fresh decoder's. A
  dense model's decoder holds no row, so its part ends a token earlier, at a token both turns
  share. A context restored without its row only drafts worse there: the verify keeps the
  tokens plain decode's.
- `/v1/stats` adds `mtp_iters`, `mtp_accepted` and `refills`, and both workers report
  `restore_s` and `save_s` (the context switches' host time).

## A client that goes away

A prompt writes nothing until its first token, for minutes on a long one, so a failed write says
nothing while it is fed. A handler waiting for a request's next event therefore looks at its client
every `CLIENT_POLL_S` (0.25 s, `opentpu/serve/api.py` `_deltas`, on chat, completions and `/generate`,
streamed or not): a non-blocking `recv(1, MSG_PEEK)`. With no FIN nothing else happens and the client
sees nothing.

A FIN alone does not mean gone. A client may half-close after sending and read on, and through
`ssh -L` a client's close reaches the server as sshd's half-close, the same FIN. So from the FIN on
each look writes something a client ignores, an SSE comment or, before a plain reply's headers, an
interim `100 Continue`. A client that closed answers with a reset and a later write fails; the request
is cancelled and the prompt stops after its current budget part. A half-closed client takes them and
gets its whole reply. Measured (`tests/test_serve_api.py`, and a stand-in server reached through an
`ssh -L` forward):

| client | no FIN while it waits | at its close |
|---|---|---|
| opencode 2.2.5 (Bun's fetch): its provider `timeout`, SIGTERM, SIGKILL | yes, direct and through ssh: it does not half-close | through ssh: writes 1-3 take, the 4th fails |
| curl `-m`, a Python socket's `close()` | yes | through ssh: the 3rd or 4th write fails |
| a Python socket's `shutdown(SHUT_WR)` | no: a FIN at once | through ssh: 15 s of writes taken, its reply whole |

At this poll the real handlers, with a stand-in worker, noticed opencode's provider timeout 0.28-0.29 s
after it fired when it ran directly, and 0.5-0.75 s after through `ssh -L` (a 2 s timeout cancelled
2.28-2.29 s after the request; a 4 s timeout 4.51-4.76 s). A client that finishes was not cancelled, and
a half-close through ssh took 31 comments and got its reply whole. The slot is free when the budget
part running ends (`BatchWorker35`: up to 512 tokens, 8-11 s on the two-card 35B). What was fed stays
the slot's context, so the same prompt again feeds only the rest
(`test_an_abandoned_prompt_frees_its_slot_and_a_retry_resumes`: its tokens a fresh engine's). opencode
retries at once on a provider timeout, so the retry is queued before the abort is noticed. It still
resumes when the worker's loop next comes round after the notice, its usual order: the loop ends the
aborted request before it binds the new one. By the reading of that loop, not a measurement, a loop top
inside the notice (0.75 s of an 8-11 s part, under 1 part in 10) binds the retry to another free slot
from position 0; the fed context stays in its slot, and the next retry takes the longest.

`SO_KEEPALIVE` with a minute to failure covers a machine that vanishes with no FIN (Linux's
`TCP_KEEPIDLE` and friends; `KEEPALIVE`). Through `ssh -L` the socket is sshd's on this host, so its
`ClientAliveInterval` is what sees the far machine.

## Measured on the card

Run on 2026-10-09 on the card host with build 1025907f (133.33 MHz, Gen2 x8), Qwen3-0.6B int8, cap
2048, using `tools/serve_bench.py`. Each client streams a greedy chat of 128 tokens
(`ignore_eos`) unless the row says sampled. Three trees: requests in turns (70ec5139), batched
with a program compiled per step (f99acce6), batched with span programs and feed runs
(6c3a1c75, with the feed-kind fix of 40fa0062).

| clients | in turns | batched, per-step programs | batched, span programs | TTFT median / max (span) |
|---|---|---|---|---|
| 1 | 25.0 tok/s | 23.9 tok/s | 24.6 tok/s | 0.24 / 0.24 s |
| 2 | 24.2 | 41.9 | 44.4 | 0.23 / 0.37 s |
| 4 | 24.0 | 53.3 | 77.7 | 0.56 / 0.56 s |
| 8 | 23.8 | 50.1 | 75.7 | 1.19 / 1.78 s |
| 8, sampled (temperature 0.7, top-k 20, top-p 0.8) | | 46.3 | 78.5 | 0.60 / 0.60 s |

- **Correctness.** Four prompts alone, then together. Every reply is equal to its own, and a
  repeat too (`check: PASS` in every run).
- **In turns.** About 49 ms per context switch.
- **Batched steps** (`tools/batch_bench.py`, 8 slots, 16 steps per case): the card's run of a
  step against its wall time. Per-step programs left the card idle while the host compiled
  the next one (a 4-row step: 44 ms on the card, 71.5 ms wall at context 64; 58.5 and 136 ms
  at 1024). With span programs:

  | rows | context 64: tok/s | card / wall per run | context 1024: tok/s | card / wall per run |
  |---|---|---|---|---|
  | 1 | 24.1 | 39.5 / 41.5 ms | 21.8 | 43.1 / 46.0 ms |
  | 2 | 45.1 | 41.6 / 44.3 ms | 37.4 | 48.8 / 53.4 ms |
  | 4 | 79.1 | 46.3 / 50.6 ms | 56.7 | 60.6 / 70.5 ms |
  | 8 | 79.2 | 91.3 / 101 ms (one run of 8) | 54.7 | 121 / 146 ms (two runs of 4) |
  | 4, sampled | 77.1 | 46.8 / 51.9 ms | 57.1 | 61.2 / 70.1 ms |

  The decode loop alone (`generate_card`) makes 25.3 tok/s at context 64. Eight rows in one
  run cost two runs of four on the card (at context 1024 their program no longer fits IMEM,
  and they run as two): the MXU takes MCOLS = 4 rows per pass of the
  weights, so four rows a run is the most a pass gives (~1.45 M cycles a token at a short
  context). A trace of a 4-row step's first 250 k cycles shows the MXU busy 95%. The wall
  time over the card's at context 1024 is mostly each case's first program, compiled inline
  and paid over only 16 steps (a program serves up to 256).
- **Prefill.** Feed runs fed 84 tok/s of 64-token prompts and 93 tok/s of 1024-token ones,
  on any slot (slot 0's prompt runs: 47.5 and 65 tok/s).
- **A conversation's turns** (each re-sends the history, max 128 tokens a reply), against the
  same prompt with nothing saved:

  | turn | prompt tokens | cached | TTFT | TTFT cold |
  |---|---|---|---|---|
  | 3 | 157 | 121 | 0.46 s | 1.56 s |
  | 4 | 196 | 153 | 0.50 s | 2.02 s |

  Qwen3's template renders a past reply without the empty thinking block its generation prompt
  had, so each turn feeds the last reply again (as otpu-chat).
- **A KV store of 50 MB of RAM** (`--kv-ram-gb 0.05`, tree 0e2d8cf6). Sixteen users, each
  with two turns of up to 64 tokens: every user's first turn at once, then every second turn
  at once (`serve_bench --users 16`). The store spilled 26 contexts to disk (260 MB in 54
  files), and 14 second turns found their first turn's context on disk, 2 in RAM (18 ms of
  disk reads in all). Three second turns run again with the cache flushed gave the same
  replies, so the contexts came back bit for bit (`users: PASS`, `check: PASS`). A second turn
  reuses only its first turn's prompt (22 of ~105 tokens): Qwen3's template renders the past
  reply without the generation prompt's empty thinking block, so the reply is fed again.
- **Requests coming and going** (16 users, two turns each, 8 slots): their row sets change
  at every join and leave. In tree 0e2d8cf6 the 16 users compiled 74 of 129 step programs
  inline (4.9 s of waits in 62 s). With rows in slot order and blocks, feed programs shared by
  every prompt (one per slot and attention block), and the coming steps compiled ahead, the
  whole serve_bench (staged tree 6e591560, 2026-10-09 20:17) compiled 14 of 86 inline, 1.08 s
  of waits in 124 s; `users: PASS`, every other part as before (4 / 8 clients 77.6 / 74.8
  tok/s, 8 sampled 81.1).
- **Models in turns** (staged tree 6e591560, `--levels 1,4`, a 3-turn conversation): every
  check passed.

  | model | 1 client | 4 clients | turn 3: cached / prompt tokens, TTFT (cold) |
  |---|---|---|---|
  | LFM2.5-230M | 64.3 tok/s | 61.5 | 217 / 232, 0.08 s (0.75 s) |
  | Qwen3.5-0.8B (DeltaNet state) | 18.7 | 17.6 | 168 / 315, 3.53 s (4.32 s) |

  LFM2's template renders a past reply as it was generated, so a turn reuses all but its new
  message. Qwen3.5's state lets a turn continue only a saved context's end: the reply's start,
  where the next turn's rendering parts.
- **Clients arriving one by one** (`serve_bench --arrivals 4`: four clients 1.5 s apart, 128
  tokens each; the gaps are between a client's streamed tokens). A prompt's feed runs hold the
  decoding requests for up to `--chunked-prefill-size` tokens (~45 ms a run of four).
  Mixed runs (`--mixed`) cut that pause but feed the prompt only in the spare rows, and they
  starve the prompts when most slots are being fed. Measured with the whole bench on a fresh
  server each (staged tree peb89a31, 2026-10-09 21:06), blocks packed:

  | | gap p50 / p99 / max | TTFT | 4 clients at once | 16 users' second turns: TTFT median |
  |---|---|---|---|---|
  | default | 45 / 302 / 345 ms | 0.30-0.39 s | 78.3 tok/s | 7.6 s |
  | `--mixed` | 45 / 104 / 145 ms | 0.32-1.21 s | 70.9 tok/s | 12.3 s |

  Before a request took a slot in the busiest block (staged tree m11b6d45, `--mixed`), the
  same part run after the others had its gaps' median at 81 ms, two runs' time: the earlier
  parts had left the least recently used slots in both blocks. Packed, each of the part's 184
  steps took one run.
- **Host time per run** (tree 0e2d8cf6, 32 steps a case): the io write 0.2-0.3 ms, the start
  0.06 ms, the picks' read 0.1-0.2 ms; the wait is the card's time within 0.1 ms. A 4-row
  step: 79.8 tok/s greedy, 79.1 sampled.

## Tests

All on the ISA simulator with tiny models.

- `tests/test_serve_kvio.py`: save and restore bit for bit (Qwen3, LFM2, Qwen3.5 on one and two
  slices, direct and staged), the state staged in batches smaller than it and read from a
  snapshot, partial restores, shared pages.
- `test_serve_store.py`: the tiers, pins, restart, the prefix cache; a prefix restored twice
  outlasting newer contexts (in RAM and in the entries' limit), and on disk at once, found
  boosted by a store started without the first's shutdown.
- `test_serve_scheduler.py`: requests in turns equal their fresh runs, prefixes from the device
  and the store, stop strings, cancels, a store on disk across workers; a chat's reply start
  kept as a snapshot, its next turn rolled back to it with no save, the idle save, a request
  restored from it, a flush; a next turn sent the moment its reply ends pre-empting the idle
  save (no queue wait, nothing stored), another conversation waiting for it; a tool call's
  reply kept on the device past the idle save, the agent's next request feeding only the
  tool's result; every reply kept (`keep_reply` `""`, the reply start at the prompt's last
  position): the next turn feeding only what is new, one whose reply part differs restoring the
  reply start, a switch before the idle keeping the reply in the store.
- `test_serve_prefix.py`: a new conversation restoring another's system prompt at its mark and
  feeding only its own turn, one whose system prompt's tail differs restoring the last block,
  tokens equal to fresh runs; the cuts; the API's mark at the final user turn's `<|im_start|>`
  (in Qwen3.6's tokens with `OTPU_QWEN36_TOKENIZER` or `models/Qwen3.5-0.8B`).
- `test_batch_steps.py`: greedy rows each sequence's decode, sampled rows the decode loop's pick,
  IMEM groups, span programs across attention blocks, feed runs then the last token's pick, a
  joining row's program compiled ahead, a leaving row's slot padding the live program, a feed
  row's kind from the compile process, mixed runs (a prompt fed beside decoding rows).
- `test_serve_batching.py`: requests in slots equal their fresh runs, a prefix copied between
  slots and restored from the store, a seeded sampled request the same alone and batched, the
  coming steps' rows told ahead (with and without mixed runs) as the loop then takes them,
  requests packed into a block of slots.
- `test_serve_api.py`: the HTTP API end to end with concurrent clients, 64 connections at
  once, SGLang's `/generate`; a client gone while a prompt is fed (FIN or reset, each route,
  streamed or not) cancelled, a half-closed one served, one that stays seeing nothing, keepalive.
- `test_serve_batching35.py`: requests in slots as above, an abandoned prompt on the live path (its
  slot free after the part running, a retry feeding only the rest, tokens a fresh engine's), and
  `progstate.warm_reach` against the files the engine and `warm35x2` write.
- `test_serve_lanes35.py`: at x2 (S = 4), every slot's save restored into every slot (4 x 4, a
  partial last page, over junk, and with the destination's whole pages left in place), fed to the
  end, is that slot's own feed from 0: logits, K and V bytes as stored, DeltaNet states and
  convolution windows; after a restart from disk, into another slot; `relane` against
  `quant.hash_rows`; `copy_from`; a store under the old key not read.
- `test_serve_prefix35.py`: at x2 (S = 4), a prefix restored into another slot and fed to the end is
  the same logits, pages, DeltaNet states and convolution windows as the prompt fed from 0, also for
  prompts that part right after a cut; new sessions (no step compiled in the restoring slot) and a
  next turn restoring; a save and a restore
  beside decoding slots; a prefix outliving a killed server; a client leaving, or a read failing, at
  a cut.
- `test_serve_tools.py`: tool calls with Qwen3.6's template (`tests/data/`): the tools
  rendered (`tool_choice` none: not), calls parsed by type, the stream's chunks in order, null
  content and tool messages accepted, the next request's prompt continuing the last prompt and
  its reply (in Qwen3.6's tokens with `OTPU_QWEN36_TOKENIZER` or `models/Qwen3.5-0.8B`); with
  `--preserve-thinking` an opencode-shaped session feeding only the tool's result and the
  user's next message (with thinking: the reasoning sent back; every reply kept, its start at
  the prompt's last position), and without it the next message parting at the first turn's
  reply.
- `test_serve_dashboard.py`, `test_serve35.py`: the monitor's routes and the 35B's path
  (docs/dashboard.md).
- `test_serve_mtp.py`: `--mtp` on the tiny streamed MoE with its MTP layer: requests in turns
  equal plain decode's, each switch back the same MTP context (the MTP layer's KV cache, the
  state, the draft), a pending draft run before the save; a prompt from a held reply start
  appends the MTP layer's row again, bit for bit a fresh decoder's (without the held row, it
  differs there and the tokens do not); a loop halted on the device mid-run continues (the
  host's stop word written at a WAITW while the ISA simulator runs it); a dense Qwen3.5.
- `test_minicpmv.py`: the worker with images (requests in turns equal their fresh runs, a
  context reused past an image only for the same image, a text-only request after them), the
  API's image_url parts (placeholders, keys, refusals).

# The card-to-card link (bcu1525-x2)

Two BCU1525s joined by two QSFP28 cables run one model in tensor parallel (docs/x2.md). This page
is the link under it: what goes over the cables, the port the collectives hub sees, its registers,
its simulation models, and what was measured on the cards. Labels: **card** is a run on two
BCU1525s, **sim** a Verilator run, **synth** a Yosys run, **estimate** arithmetic.

## Summary

- **The wire.** Each QSFP28 cage carries one Aurora 64B/66B channel of four GTY lanes at
  25.78125 Gb/s, streaming mode, no Aurora flow control or CRC. The framing, the credits and two
  CRC-32s are the port's own.
- **Errors (card).** IBERT: 0 errors on all 16 lane-directions in 300 s (7.84e12 bits each, BER
  < 3.83e-13 at 95 %). Aurora, both cables both ways for 10 minutes: 0 errors in 1.98e14 bits
  (BER < 1.5e-14 at 95 %), no relock, no channel drop.
- **Rate (card).** 88.6 Gb/s of payload a cable and direction, 11.1 GB/s. The hub's GATHER moves
  at most 4.8 GB/s (8 words a cycle at 150 MHz), so one cable alone keeps up.
- **Latency (card).** 205-208 ns from one card's port to the other's, idle: 31 core cycles at
  150 MHz. The budget from one card's hub to the other's is about 39 cycles one way (section 7;
  a budget, not a measurement).
  On the two-card models that costs at most about 0.2 % of a token (docs/x2.md).
- **Area (synth).** The generic endpoint, two ports: 19.9K LUT sites, 8.8K flip-flops, 8 RAMB36
  and 58 RAMB18, without the Aurora cores and the transceivers.
- **What it does not have.** No link-level replay: a message that fails a CRC flags its
  collective, and the host runs the token again (docs/x2.md, "Errors and recovery"). Forwarding
  for a ring of more than two cards is specified (section 4) and not built.

## 1. Pieces

| Piece | Where | What |
|---|---|---|
| `otpu_link` | `rtl/link/otpu_link.sv` | the generic endpoint: two links, each a core-clock port (`otpu_link_port`), a user-clock lane side (`otpu_link_lane`: framing, control words, CRC, hello, ping), a 512-word TX FIFO and a 2,048-word (64 KB) dual-clock RX FIFO (`otpu_link_cdc.sv`) |
| `otpu_link_pkg` | `rtl/link/otpu_link_pkg.sv` | the header's fields and the message types |
| `otpu_link_phy_bcu` | `boards/bcu1525/link/` | the two Aurora cores, the GT common blocks, the reference clock and the reset release |
| the hub's adapters | `otpu_coll_lx` in `rtl/top/otpu_coll.sv` | one a link: the crossing between the hub's die and the link's (XRS register stages each way, a TX skid, RX pulled against the hub's credits). No handshake crosses the dies |
| the registers | `rtl/shell/otpu_ctrl.sv`, 0x300-0x3FF | the X2 window (section 5) |
| `otpu_link_sim` | `sim/verilator/` | the cable model of the two-board co-simulation (section 6) |
| `otpu_link_pair`, `otpu_link_chan` | `sim/verilator/` | two real endpoints over an Aurora stand-in, with `otpu_link_sim`'s ports |

Port 0 is QSFP0 (GT quad 231), port 1 is QSFP1 (quad 230). Both sit in SLR2, beside the quads.

## 2. The port

Per link, AXI-Stream ends in the core clock:

| signal | direction (port's view) | meaning |
|---|---|---|
| `s_tdata[255:0]`, `s_tvalid`, `s_tlast` / `s_tready` | in / out | TX: a message is a header word, then LEN payload words; `tlast` on the last |
| `m_tdata`, `m_tvalid`, `m_tlast`, `m_thdr` / `m_tready` | out / in | RX: messages as they arrive; `m_thdr` on the header |
| `up` | out | level: the channel and all lanes up, and a hello received that agrees (the peer's NCARDS mine, its card id not mine) since the last change of X2_CARD or link reset |
| `rx_err` | out | pulse: a message failed HCRC, CHK or LEN, its trailer's DCRC failed, the RX FIFO overflowed, or the channel dropped while up |
| `abort_in` | out | pulse: the peer sent ABORT |
| `abort` | in | pulse: send ABORT at the next message boundary (no credits needed) |

- **Credits.** The port takes a header only when the peer's RX FIFO has room for the whole
  message (LEN + 1 words, plus a trailer when LEN > 0). A message never waits for credits once
  begun, so two links cannot hold each other up mid-message. The window opens at 2,040 words,
  eight words of guard in the 2,048-word RAM. Returns are cumulative modulo 65,536, and a missing
  or corrupt header returns no guessed credits: after an error, credits can stay outstanding
  until the host stops both cards and resets the link.
- **Hold.** The hub keeps `m_tready` low until its own slices have issued the collective a
  message belongs to; messages wait in the RX FIFO. The port knows nothing about collectives.
  Credits go back to the peer as the hub pops.
- **Reset.** The hub's reset (RUN = 0) does not touch the port, and in reset the hub never pops.
  A card that starts first parks its messages in the other card's RX FIFO, so start skew is
  harmless. The link reset (X2_CTRL LRST: the control block's pulse of at least 16 core cycles,
  ORed with the board reset) clears this end's FIFOs, credits, counters, ping result and
  stickies. The host pulses it on both cards together, with both cards stopped. It does not
  retrain the transceivers.

## 3. Messages

- **The header word**, low 64 bits: TYPE, FLAGS, LEN, ARG, AUX, CHK.
- **The hub's DATA message** (TYPE 0x44, LEN <= 32):
  - ARG is the collective's sequence number (16 bits);
  - AUX is 1 for a GATHER piece, 2 for a BAR token (a header with LEN 0); 3 is reserved for the
    ring (section 4);
  - [71:64] the global slice, [79:72] the origin card, [83:80] HOPS, [111:96] the row and
    [127:112] the column of the first payload word;
  - the payload is 8 x 32-bit TMEM words a word, lane l in [32l + 31 : 32l].
- **What the port adds and strips** (the hub sends zeros there and never sees them):
  - FLAGS and CHK;
  - **HCRC**, header [255:224]: CRC-32 of [223:0] (MSB first, initial state 0xFFFFFFFF,
    polynomial 0x04C11DB7, no reflection, no final XOR). It is checked before the header is
    shown on `m_*`. A bad header raises `rx_err` and is not delivered; the parser then scans for
    a valid known header;
  - **the trailer**, one word after the last payload word (LEN > 0): [31:0] DCRC, the same CRC
    continued over the payload words, and [63:56] 0x54. The receiving port checks it as the hub
    pops past the last payload word and drops it. On a mismatch `rx_err` pulses once the data
    has been delivered: the data is not held back, the token is invalid.
- **Control words** (CRED, IDENT = hello, ABORT, PING / PONG) go out between messages, never
  behind more than one, and take no credits. The receiving port consumes them. Hello goes out at
  channel up, after every change of X2_CARD, on X2_CTRL HELLO, and every 2^21 user-clock cycles
  (5 ms).
- **Streaming with the port's CRCs, not Aurora's framing.** Streaming is the mode the cards
  measured error-free. Aurora's soft errors go to counters, and the CRCs catch the data they
  corrupt; a hard error drops the channel, which raises `rx_err` and drops `up`.
- **Priority.** A message goes out cut-through as the hub writes it. A 32-word message fed at a
  word a core cycle holds its link for about 33 core cycles (220 ns), so a control word can wait
  that long. At two cards only CRED, hello, ABORT and PING share a link with the hub's messages,
  and none is latency-critical: the RX FIFO is 64 KB against a credit loop of about 140 words.

## 4. A ring of more than two cards (specified, not built)

- **Forwarding.** With HOPS > 1 the port would forward the message out of its other link with
  HOPS - 1, cut-through in the user clock, and deliver it locally too. At two cards the hub
  always sends HOPS 1.
- **The order rule.** A link carrying both forwarded and local messages must keep the
  collectives' order (ARG): a forwarded k + 1 ahead of a local k on the same cable would have the
  far hub hold that link at k + 1 while it needs the k behind it. The port would order by ARG,
  learning that the hub's local traffic for k is done from its first k + 1 header or from an
  END marker (AUX 3, LEN 0).
- **Cabling.** From four cards on, port 0 of card c goes to port 1 of card c + 1; the host checks
  it from the hellos. At two cards either cabling works.
- **Latency (estimate).** A forwarded hop is the Aurora cores, the cable and a few stages, about
  0.155 us, so four hops are about 0.26 + 3 x 0.155 = 0.73 us.

## 5. Registers (the X2 window, 0x300-0x3FF)

`opentpu/host/regs.py` names them. CAPS2 bit 15 (XCARD) says the build has the window; the
platform id of bcu1525-x2 is 7.

| offset | register | bits |
|---|---|---|
| 0x300 | X2_CARD | [7:0] CARD_ID, [15:8] NCARDS. Reset 0 / 1: a single card |
| 0x304 | X2_CTRL | self-clearing: [0] clear stickies, [1] hello now, [2] LRST (link reset), [3] ping |
| 0x308 | X2_INFO | read-only: [7:0] PROTO, [15:8] LMAX 32, [31:16] the RX FIFO's words, 2,048 |
| 0x30C | X2_SEQ | read-only: collectives issued locally since the link reset, modulo 65,536. RUN does not reset it |
| 0x310 + 0x10 p | port p's status | [0] channel up, [1] hello received, [2] CRC error (sticky: any `rx_err` cause), [3] watchdog (sticky: a message waited for credits more than 2^24 cycles), [4] ABORT received (sticky), [5] all lanes up, [6] dropped while up (sticky), [7] up, [31:16] credits available |
| + 4 | hello | the far end's [7:0] CARD_ID, [11:8] port, [19:12] NCARDS, [31:24] magic 0xA5 |
| + 8, + C | | CRC errors; frames received |
| 0x330 + 4 p | ping | port p's last round trip in core cycles, [31] valid |
| 0x340 + 0x10 p | counters | + 0 words sent, + 4 words received, + 8 credit-wait cycles, + C soft errors |
| 0x360 | COLL_STATUS | read-only: [0] aborted, [1] watchdog, [2] a header out of place, [3] link check, [4] ABORT received, [6:5] links up |
| 0x364 | COLL_WD | the hub's watchdog in core cycles; 0 is off, reset 2^26 |

The board owns X2_CARD, X2_SEQ, COLL_STATUS and COLL_WD; the other offsets pass through to the
link's own decode. Clear-stickies does not erase the counters or the last ping, and does not
re-enable a hub that aborted: that takes a link reset on both stopped cards. A build with one
card (NC 1) has no link logic and the window reads zero.

**The host's check at open** (`opentpu.host.cards.CardsTransport`): write X2_CARD on each card,
pulse LRST on both, wait for both ports of both cards to report up, then read the hellos. Each
must carry the magic and name the other card; the cable map must be symmetric. A missing or
asymmetric link stops the run with the registers in the message.

## 6. Simulation

- **`otpu_link_sim`**: one cable, both ends in the core clock with section 2's signals. It models
  a one-way latency of L core cycles (default 40; the tests use 0, 40 and 240), an RX FIFO with
  the port's credits, a word-rate cap (default 2.31 words a cycle, 11.1 GB/s, which never binds
  at a word a cycle), and nothing on the wire for control words but ABORT. Plusargs
  (`+lnk<ID>_...`) flip a bit in message n, drop message n, stall the cable from cycle t for c
  cycles, or abort. Its ping and its traffic counters read zero.
- **`otpu_link_pair`** has the same ports and holds two real `otpu_link` endpoints joined by
  `otpu_link_chan`, an Aurora stand-in at the user clock's rate (402.83 MHz): registered delay,
  clock-compensation gaps and long `tx_tready` stalls, channel and lane loss, soft errors, and
  targeted faults (a header's HCRC, CHK or LEN; a payload or trailer word; a dropped header,
  payload or trailer) in either direction on either cable. It replaces a cable of the co-sim
  without a change of connections, and it is the check of the model against the RTL.
- **`tests/test_x2_link.py`** (91 cases): every word checked at all four ends; the same random
  traffic through the RTL and the model, compared by whole-stream digests; the RX memory never
  past 2,048 words and the credits never past 2,040; a long RX hold reaching credit exhaustion;
  a rollover case of 102,000 message words a direction; the registers, ping in core cycles,
  ABORT both ways on both links, hello agreement and its changes, channel and lane loss, each
  fault's delivery, error timing and resynchronization; straight and crossed cabling; and
  `verilator --lint-only -Wall` of the endpoint and the PHY wrapper.

## 7. On the cards

Two BCU1525s, 0.5 m passive copper DACs (the modules' EEPROM says 1 m), Vivado 2026.1, a test
image of its own (the endpoint's CDC and framing, test traffic, a latency meter).

**IBERT at 25.78125 Gb/s** (300 s, DFE on every lane): 0 errors on all 16 lane-directions,
7.835-7.842e12 bits each (BER < 3.83e-13 at 95 %; all 16 together 1.25e14 bits, < 2.4e-14). Each
card's QSFP0 lane n meets the other card's QSFP0 lane n, QSFP1 likewise. IBERT's status column
read "17.188 Gbps" on every lane, 2/3 of the rate: its rate counters take their clock's period as
10 ns whatever the system clock is, and the image ran them at 150 MHz. The bit counts give
25.8 Gb/s. Feed IBERT's system clock exactly 100 MHz.

**Aurora, idle latency** (100,000 pings on each cable and direction, 0 lost, the eight
measurements within 1 ns):

| | round trip (core clock) | one way |
|---|---|---|
| turned round in the far end's user clock | 54.9 cycles, 366 ns (347-400) | Aurora TX + RX and the cable: 144 ns |
| turned round in the far end's core clock | 62.5 cycles, 416-417 ns (393-453) | port to port: 208 ns |

- Aurora and the cable are 144 ns (the near-end PMA loopback, with no cable: 139-140 ns). The
  port's crossings and stages are 30.3 ns at each end, 61 ns in all (sim: 56 ns). Port to port:
  2 x 30.3 + 144 = 205 ns, 31 core cycles at 150 MHz.
- With the hub (its adapters' stages between the dies, the header's CRC check), the budget one
  way at 150 MHz and XRS 2:

| step | core cycles |
|---|---|
| hub, XRS stages, adapter, the port's TX FIFO | 3 |
| core clock to user clock | 4.5 |
| Aurora TX, the cable, Aurora RX | 21.6 |
| user clock to core clock (the RX FIFO) | 4.5 |
| HCRC check | 1-2 |
| adapter, XRS stages, hub | 3 |
| **one way** | **about 39** |

- Loaded (the soak's traffic on every link): a ping's round trip is 84-94 cycles against 55
  idle. It waits behind the message in flight at each end, as the simulation predicted.

**The soak** (10.0 minutes, every link both ways at once: 64-word PRBS messages, 37-beat data
messages checked end to end, pings):

| receiver | PRBS words | bit errors | BER (95 %) | relocks, soft, hard, drops | data messages good / bad | RX rate |
|---|---|---|---|---|---|---|
| card A QSFP0 | 1.936e11 | 0 | < 6.05e-14 | 0 0 0 0 | 378.2 M / 0 | 90.78 Gb/s |
| card A QSFP1 | 1.937e11 | 0 | < 6.05e-14 | 0 0 0 0 | 378.2 M / 0 | 90.67 Gb/s |
| card B QSFP0 | 1.937e11 | 0 | < 6.05e-14 | 0 0 0 0 | 378.2 M / 0 | 90.67 Gb/s |
| card B QSFP1 | 1.937e11 | 0 | < 6.05e-14 | 0 0 0 0 | 378.2 M / 0 | 90.60 Gb/s |

- All four together: 0 errors in 1.98e14 bits (BER < 1.5e-14 at 95 %), no header dropped, no
  FIFO overflow. Dies 31-34 C.
- The RX rate counts every word Aurora delivers. The payload is about 0.859 of the user clock's
  cycles: 88.6 Gb/s, 11.1 GB/s a cable and direction, 22.1 GB/s a card and direction with both.
  Streaming Aurora's ceiling is 64/66 of the line, 12.5 GB/s; the rest is the TX's message
  boundaries and its skid behind `tready`.
- Even at the hub's full 4.8 GB/s without pause, the soak's bound allows about two bit errors an
  hour; decode moves a small fraction of that.

**The reference clock.** The card's two QSFP reference oscillators (SI5335) have reset and
frequency-select pins on the FPGA. Images that drove those resets from a counter on the board's
300 MHz clock came up with no JTAG debug hub: the oscillator behind QSFP0 runs locked to the
300 MHz clocks, so resetting it from them deadlocks. The link drives no SI5335 pin. Both quads
take QSFP0's 161.1328125 MHz reference (quad 230 over the south-bound reference path), and the
PHY's reset release counts on the configuration clock (STARTUPE3 CFGMCLK), not on a board clock.
A release of the SI5335 resets from CFGMCLK also works (measured: each quad then has its own
161.13 MHz reference); the link does not need it.

**In the two-card image** (docs/x2.md): every run ends with the hubs' sequences equal and no CRC
or soft error on either port, through the qualification's collectives and the models' cuts
reported there (the chat sessions' records do not state the link counters).

## 8. Area

`otpu_link` alone, both ports, with no Aurora core, transceiver or I/O buffer. Yosys 0.68 with
the slang front end, `synth_xilinx -family xcup -top otpu_link -noiopad -abc9`:

| resource | cells |
|---|---:|
| LUT2 .. LUT6 | 19,597 |
| SRL16E | 254 |
| flip-flops | 8,794 |
| RAMB36E2 | 8 |
| RAMB18E2 | 58 |

Also 382 CARRY4, 5 MUXF7, 1 MUXF8. This is a synthesis estimate with no placement or timing, and
Yosys counts wide multiplexers about 2x lower than Vivado does, so the LUTs are not a Vivado
utilization.

## 9. Open

- The ring: forwarding, the order rule and the END marker (section 4), and a projection of the
  models at more than two cards (the models here do not split 16 ways).
- No link-level replay. Phase 1 is a CRC a message, the collective flagged, and the host running
  the token again.
- A PING under load waits behind a whole message. If control words ever become latency-critical,
  shorter messages or word-level priority would fix it, at a cost in tail latency.

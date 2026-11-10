# BCU1525: the System Management Wizard's temperature bus can stop

On the BCU1525 the thermal governor (`rtl/boards/bcu1525/otpu_bcu_gov.sv`, docs/bcu1525.md
section 14) takes the die's temperature from the `temp_out` bus of AMD's System Management Wizard
(system_management_wiz v1.3 rev 25, Vivado 2026.1). That bus can stop for good, holding its last
code, after an AXI read or write of any of the wizard's DRP registers: about once in 994 such
accesses. This page is the root cause, the simulation that reproduces it on the generated files,
what the governor and the host do about it, and what would remove it.

**Short answer.**
- The wizard's DRP arbiter loses the temperature bus's own read when an AXI DRP request lands in
  the first cycle of the bus FSM's busy. The FSM then waits for a DRDY that never comes, with no
  timeout. Host reads keep working, so nothing looks broken.
- The governor now treats a die code that has not changed for 60 s as no reading
  (`TEMP_STALE_MS`, STATUS cause bit23, TEMP_AGE at 0x48): RUN ends and GO waits. The SYSMON's
  alarm and over-temperature lines do not come through the bus and stayed live throughout.
- The host tools read no DRP register by default. A frozen bus is restarted, in SAFE, by the
  wizard's SYSMONRR.
- The race itself is still in the image. Three ways to remove it without editing the generated
  IP are listed below; none is built.

## What the card showed (2026-10-29)

- On one WG image the governor's TEMP read code 656 (46.05 C) for about 3.5 h, until the image
  was replaced. `otpu_ctrl`'s R_TEMP, fed by the same bus, read the same. Over that time the
  SYSMON's own temperature register (BAR0 0x30400) read 34, 76 and 40 C, and AXI reads of the
  SYSMON kept working.
- The next image, with the same block design, tracked the register in RUN (42.07 / 43.56 /
  44.56 C on both readings over 30 s). The freeze is intermittent.
- In the session log the freeze starts in the second in which the bring-up script read five
  SYSMON DRP registers over BAR0 (0x400, 0x540, 0x550, 0x54C, 0x55C). Nothing had read those
  registers over AXI before: the fan loop read the governor's TEMP only, and Vivado's JTAG monitor
  uses the SYSMON's JTAG port, not this logic.

## The design

`boards/bcu1525/vivado/bd_wg.tcl` builds the wizard with `ENABLE_TEMP_BUS`; its `s_axi_aclk` is
core_clk and its reset is the core's (PERST# and the clock's lock: the governor's GO does not
touch it). `temp_out` feeds the governor and the control block's R_TEMP.

Inside the wizard (`xadc_core_drp.vhd`) two requesters share the SYSMONE4's DRP through
`drp_arbiter`:

- **Port A** is `temperature_update`. It refreshes `temp_out` with a DRP read of address 0x00.
- **Port B** is the AXI-Lite path. A read or write in the DRP range (0x400-0x7FC) is one request,
  a single-cycle DEN. Port B's busy input is tied to 0.

`temperature_update.vhd` is a four-state FSM:

| State | What it does |
|---|---|
| WAIT_TIMER_UPDATE | leaves when the timer is 0..8; the timer counts down from wait_cycle (0x3E8 from reset) and reloads |
| WAIT_UPDATE_ENABLE | `busy_o <= '1'` |
| RD_EN_CTRL_REG_TEMP | `den_o <= '1'` for one cycle, address 0x00 |
| RD_CTRL_REG_TEMP | waits for `drdy_i`, with no timeout; then `temp_out <= do_i(15 downto 6)` and busy drops |

So each refresh raises busy for one cycle (B1) before its one-cycle DEN (B2). A refresh starts
every 994 cycles in simulation: 6.6 us at 150 MHz.

`drp_arbiter.vhd`, state NOGRANT with `bbusy_A = '1'`:

1. **Cycle B1** (A busy, no DEN yet). An AXI request (den_B) is parked: `overlap_B <= '1'`, its
   address saved. Nothing is granted this cycle, although the DRP is free.
2. **Cycle B2** (den_A). The process first grants A (`daddr_C_reg <= daddr_A`, GRANT_A), then,
   further down the same branch, sees `overlap_B = '1'` and overwrites that grant with B's
   (`daddr_C_reg <= daddr_reg`, GRANT_B). The last assignment wins.
3. **A's request is lost.** It is neither issued nor saved: nothing in this branch sets
   `overlap_A` for den_A, as the mirror branch and GRANT_B do.
4. **GRANT_B** serves B and its DRDY goes to B. A waits in RD_CTRL_REG_TEMP forever with busy at
   1, and `temp_out` keeps its last value until the wizard's reset.
5. **Host reads still work.** With bbusy_A stuck at 1 every later AXI request takes the same
   branch: parked one cycle, served the next. This is what the card showed.

**Odds:** one bad cycle per bus period, about 1 in 994 per AXI access of a DRP register.

| Reader | Chance of freezing the bus |
|---|---|
| a bring-up that reads five DRP registers | 0.5 % a run |
| a fan loop reading 0x30400 every 2 s | once in about 33 min |
| a monitor with 13 DRP reads a query | 1.3 % a query; once a second, about 76 s on average |

## Reproduced in simulation

`boards/bcu1525/sim/tb_wiz_temp_bus.vhd` runs the two generated files unchanged under GHDL (they
are AMD's and not in this repository: a WG build's `otpu.gen` directory has them; the commands are
in the testbench's header), with a model DRP (DRDY 4 cycles after DEN).

- One port-B request lands K cycles after the bus FSM raised busy, for each K from 0 to 1010,
  which covers every phase of the 994-cycle period. Then the die's code changes.
- **K = 0** (busy's first cycle, before its DEN): `temp_out` keeps the old code, busy stays at 1,
  and B's next request still completes.
- **Every other K:** the bus follows the new code.

Output: `the bus FSM's period: 994 cycles`, `K=0 frozen: temp_out 600 busy '1' B done 2`,
`trials 1011, frozen 1`. The result does not depend on the DRP's latency.

## Why not JTAG, GO or a reset

- **JTAG.** Had a JTAG DRP access swallowed the bus's DEN, the arbiter would wait in GRANT_A
  forever and every AXI read of the SYSMON would hang; they kept working. JTAGLOCKED, which
  Vivado's monitor does not set, parks both ports and serves them after.
- **GO.** GO releases the DIMMs and the accelerator. The wizard's reset is the core's, which GO
  does not touch. The freeze coincided with GO only because the bring-up read the SYSMON just
  before its first GO.
- **A reset.** SYSMONRR (0x10, bit0), SRR (0x00 = 0xA) or the core reset restarts both FSMs. A
  reset is a recovery, not a cause. `temp_out` reads 0 until the next refresh, under the
  governor's VALID_MIN, so the governor drops TEMP_VALID and goes SAFE.

## What a frozen bus costs the governor

The governor samples the bus (two agreeing samples) and judges TRIP, WARN and RESUME from that
code. Its two alarms do not come through the bus: `user_temp_alarm_out` is the hard block's
ALM[0] and `ot_out` its OT. With the bus frozen:

- **Still live:** the temperature alarm (80 C up, 70 C down) and OT (84 C). Each is a SAFE cause
  and refuses GO. TRIP_MAX is 79.9 C, so the alarm stands in for TRIP within 0.1 C.
- **Lost:** `cool`, and with it GO's permission: the frozen 46.05 C read as cool, under RESUME
  55 C, while the die was at 76 C. Also WARN (the LED), the maximum, and the host's view of the
  die (the fan loop, otpu-smi).

## The governor's staleness check (in the RTL)

A frozen bus cannot be told from a steady die by its value. Time tells them apart: a live die's
code moves in 0.5 C steps, and 8 h of the fan loop's log from the card has no identical run
longer than 14 s. So a code that has not changed for `TEMP_STALE_MS` (60 s) is no reading:

- `otpu_bcu_gov.sv` counts milliseconds since the sampled code last changed (17 bits,
  saturating). The count holds, without clearing, while the temperature alarm stands: a bus that
  froze while the die was hot gets no fresh 60 s when the alarm clears.
- A stale code drops TEMP_VALID. RUN ends with STATUS cause bit23 (temperature stale, a thermal
  cause; a code out of range keeps bit20) and GO waits until the code moves.
- TEMP_AGE (0x48: bits 16:0 the age in ms, bit31 stale) and TEMP_STALE (0x4C: the build's limit,
  0 on builds before it) show it to the host, which can then tell "the same for 59 s" from "just
  changed" without keeping history.
- The price of a false stale is a refused GO or an ended RUN when a die truly holds one code for
  60 s; the card's data says that does not happen. A missed freeze cost a 3.5 h blind RESUME
  check and a stopped fan.
- 27 flip-flops and a comparator. `tb_bcu_gov` covers it: a held code drops TEMP_VALID and ends
  RUN, a code that changes once in each window never does, the count holds under the alarm.

This bounds the damage. It does not remove the race: the bus still freezes, and only SAFE plus a
host reset brings it back.

## The host side

- `scripts/fan_pid.py` reads 0x30400 only once the governor's die has read the same for 60 s or
  is not valid. By then its reads cannot make things worse.
- `otpu-smi` (0.4.1) follows the same rule; its `--sysmon` (13 reads a sample) is opt-in.
- `scripts/wg_bringup.py` and `scripts/fullimage_jtag.py` read no DRP register. Their die is the
  governor's code, its alarm and OT lines, TEMP_AGE where the build has it, and the SYSMON over
  its JTAG port.
- **Recovery, in SAFE only:** SYSMONRR (BAR0 0x30010) bit0, written 1 then 0. TEMP drops to code
  0 until the next refresh, so in RUN the governor would go SAFE on it.
- **Optional:** CONVSTR (0x3000C) holds the bus's wait cycles in bits 17:2. Writing 0xFFFF << 2
  makes the bus refresh about every 65.5k cycles (0.44 ms), and the odds per host read drop
  about 66 times, to about 1.5e-5. CONVSTR is a local register, not a DRP access, so the write
  cannot trigger the race; a reset sets the period back. Not used by the tools.

## Removing the race (not built)

The race needs two DRP requesters. Taking either away removes it, in the block design or in our
RTL, without editing the generated IP. Timing the host's requests does not work: the FSM's phase
cannot be seen from outside the IP (`busy_out` is the hard block's BUSY, `temp_out` changes only
when the code does, and the period shifts whenever the FSM's read waits behind a port-B read).

- **a. The governor reads the die over AXI and nothing uses `temp_out`.** A small AXI-Lite read
  master in the governor, one read of 0x30400 a millisecond, as a second master on the control
  SmartConnect, which hands the wizard one whole transaction at a time. The bus FSM may still
  run and still stop, harmlessly: with nothing reading `temp_out`, a stuck bbusy_A only parks
  each port-B request one cycle. Keeps every reading for the host. About 100 LUTs and a
  SmartConnect slave port.
- **b. Keep `temp_out`, keep the host off the DRP.** An AXI-Lite shim in front of the wizard that
  forwards the local registers (0x000-0x3FF), answers 0x400 from the governor's code and drops
  the rest of the DRP range. Port B never sees a request. The host loses the supplies, extremes
  and thresholds over PCIe (JTAG still has them). The least logic.
- **c. The wizard with its DRP interface instead of AXI, and our own DRP owner** serving the
  governor's periodic read and an AXI-Lite slave for the host, with a DRDY timeout. All of it
  our RTL, testable in Verilator; needs an IP generation to confirm that the DRP-only variant
  leaves out `temperature_update` and the arbiter.

Option a is the one to build first: it removes the race and keeps every reading.

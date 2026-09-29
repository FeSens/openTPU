# The configuration flash (draft)

Draft, 2026-09-28. Two parts: writing the card's flash over JTAG now, so that the production
bitstream loads at power-up, and a proposal for writing it over PCIe from the running design,
with a golden image and fallback. Only the lines marked **measured** were checked; nothing
here has run on the card yet. The flash still holds the card's factory image, and no backup of
it exists.

## What the board has

- **Flash:** parallel NOR, BPI x16, 64 MB. The vendor constraints
  (`constraints/vendor_ypcb003381p1.xdc`) wire flash A1..A25, DQ[15:0], CE#, OE#, WE# and ADV#
  to the FPGA's BPI configuration pins. They wire no RS[1:0], WAIT or CLK pins.
- **Configuration settings** (`constraints/otpu_top.xdc`):
  - banks at 1.8 V: CFGBVS GND, CONFIG_VOLTAGE 1.8;
  - `CONFIG_MODE BPI16`;
  - `BPI_SYNC_MODE DISABLE` (asynchronous reads);
  - `COMPRESS TRUE`;
  - `UNUSEDPIN PULLNONE`.
- **Part:** most likely a Micron MT28GU512AAA (G18: 512 Mb, 1.8 V, 256 KB blocks, a 512-word
  program buffer, the Intel/Micron command set). This is inferred, not confirmed:
  openFPGALoader's BPI support was written for this board (`ypcb003381p1` is the only BPI
  board in it), and it hard-codes that part. Vivado's name for it is `mt28gu512aax1e-bpi-x16`
  (alias `28f512g18f-bpi-x16`). Confirm it from the chip's marking or its ID register
  (manufacturer 0x0089 is Micron) before any write.
- **The pins after configuration:** these are multi-function configuration pins. Without
  `BITSTREAM.CONFIG.PERSIST` (not set; its default is off), they become user I/O once the FPGA
  is configured, so a design can drive them. Today's design leaves them unused and floating
  (PULLNONE).
- **Byte order** (**measured** on `deploy_pnbl32_e2521032`: the whole `otpu.mcs` equals this
  transform of `otpu.bit` after its header):
  - The FPGA reads 16-bit words. It takes the first bitstream byte from D[7:0] and the second
    from D[15:8], with the bits of each byte reversed.
  - The `.mcs` stores each flash word big-endian.
  - So flash word `w` = `rev(bit[2w+1]) << 8 | rev(bit[2w])`, where `rev` reverses the bits of
    a byte.
  - `opentpu/host/bpi_image.py` does the transform (it is its own inverse). It also parses what
    an image sets (IDCODE, COR0/COR1, WBSTAR, IPROG) and lists the images in a flash dump:
    `python -m opentpu.host.bpi_image info FILE`.
- **Configuration time at power-up** (computed, not measured):
  - Our images leave CONFIGRATE at its default: COR0 = 0x02003FE5, OSCFSEL 0, about 3 MHz.
  - At 16 bits × 3 MHz, the 13.1 MiB production image takes about 2.3 s after power-up
    before PCIe can train.
  - The PCIe expectation is about 100 ms. BIOSes usually enumerate later than that, but by how
    much depends on the PC.
  - The factory image, which does enumerate at boot on both PCs, shows in the backup what rate
    it uses (its COR0).
  - A faster image needs the bitstream written again from the routed checkpoint with
    `BITSTREAM.CONFIG.CONFIGRATE` (and BPI page mode: `BPI_PAGE_SIZE`,
    `BPI_1ST_READ_CYCLE`), within the flash's asynchronous read timing (UG470, "Determining
    the Maximum Configuration Clock Frequency"). That changes nothing in the design.

## Over JTAG (now)

`boards/ypcb-00338/flash_jtag.sh` uses the Vivado hardware manager: native Vivado 2026.1 on
the PC with the cable (opentpu). Its modes:

| Command | What it does | Takes the card |
|:--|:--|:--|
| `./flash_jtag.sh detect` | The JTAG chain, and the FPGA's configuration registers (BOOTSTS: where the running image came from). Reads only. | No, but it uses the cable |
| `./flash_jtag.sh backup OUT.bin` | Reads the whole 64 MB, in flash order, then prints the sha256 and `bpi_image info` of the file. | Yes |
| `OTPU_FLASH_WRITE=yes ./flash_jtag.sh program X.mcs` | Erases, programs and verifies the range that `X.mcs` covers. It refuses a file without an xc7k480t image at address 0. | Yes |
| `./flash_jtag.sh verify X.mcs` | Compares the flash with `X.mcs`. | Yes |
| `./flash_jtag.sh boot` | JPROGRAM: the FPGA configures from the flash. | Yes |

`backup`, `program` and `verify` first load Vivado's flash programming core into the FPGA.
That replaces the running design, so the card leaves the bus. Afterwards, load a bitstream or
run `boot`, then `sudo otpu-setup --rescan`.

Checked against Vivado 2026.1 without the card (on omarchy):

- the syntax of `create_hw_cfgmem`, `readback_hw_cfgmem` (`-all -format -file`) and
  `program_hw_cfgmem`;
- that the part `mt28gu512aax1e-bpi-x16` exists.

Not checked:

- whether hw_server sees the FPGA behind the card's Inspur CPLD (IDCODE 0x10931093, IR length
  8). `program.sh` gives openFPGALoader that device explicitly; `detect` shows what Vivado
  makes of the chain;
- the run times.

The session, one step at a time, with the lead's go and, for step 3, the user's:

1. `detect`.
2. `backup ~/otpu-build/factory-flash/factory-<date>.bin`. Copy it to the Mac as well. This
   file is the only copy of the factory image once step 3 runs. From it, read the factory
   image's COR0 (its configuration rate) and where its image starts.
3. `program ~/otpu-build/deploy_pnbl32_e2521032/otpu.mcs`. The image lands at address 0 and
   replaces the factory image.
4. `boot`, then `sudo -n /usr/local/sbin/otpu-rescan` (the link retrain gets its first real
   test here) or a warm reboot, then `otpu-selftest`. The config line must name build
   e2521032.
5. A cold power cycle, done by the user: does the card enumerate at power-up (about 2.3 s of
   configuration)? If not, write a version with a faster CONFIGRATE (above). Until then,
   JTAG and a rescan work as before.

Alternative: `make flash` (`program.sh --flash`) uses openFPGALoader's BPI bridge. It needs
`-b ypcb003381p1`: without the board entry, openFPGALoader 1.1.1 picks its SPI bridge and stops
("fail to open .../spiOverJtag_xc7k480tffg1156.bit.gz", 2026-09-28; the flash was not
touched). `program.sh` now passes the flag, but that is untested. openFPGALoader does not
verify BPI writes. Its `--dump-flash --file-size 67108864 OUT.bin` reads the flash in
bitstream order.

## Over PCIe (proposal)

The goal: write the flash from the running design over BAR0, with no JTAG cable and no reload,
and reboot into a chosen image with IPROG. A golden image keeps the card reachable when an
update fails to configure.

### Flash layout (64 MB = 256 blocks of 256 KB)

| Slot | Bytes | WBSTAR (words) | Holds |
|:--|:--|:--|:--|
| 0 golden | 0x0000000-0x11FFFFF (18 MB) | 0 | A qualified image with the flash controller; `NEXT_CONFIG_ADDR` = slot 1 and `CONFIGFALLBACK`. Written once (JTAG, or `otpu-flash write 0 --force-golden`). |
| 1 boot | 0x1200000-0x23FFFFF | 0x0900000 | The production image, with `CONFIGFALLBACK`. This is what runs after power-up. |
| 2 try | 0x2400000-0x35FFFFF | 0x1200000 | A candidate that `otpu-flash boot 2` loads until the next power cycle. |
| manifests | 0x3600000-0x363FFFF | - | One block: a record per slot (length, sha256, BUILD_ID, source deploy, date). |
| spare | 0x3640000-0x3FFFFFF | - | 9.75 MB. |

**Slot size.** Our compressed images are 13.1 MiB at MCOLS=2; a fuller design compresses less.
An uncompressed xc7k480t bitstream is about 18 MB; check UG470's table 1-1 before fixing the
slot size. `otpu-flash` refuses an image larger than its slot.

**WBSTAR** is what the FPGA drives on A[28:0], and here A00 goes to flash A1. So for this x16
flash it is taken as the word address (byte offset / 2). XAPP1246 uses one number for both
`NEXT_CONFIG_ADDR` and the `write_cfgmem` address, while its RS-pin table is in word units;
the first IPROG on the card settles it. A wrong address finds no sync word and falls back to
the golden image, so the mistake is safe.

**At power-up**, sources UG470 chapter 7 and XAPP1246:

1. The golden image configures from address 0.
2. Its embedded IPROG jumps to slot 1.
3. If slot 1 fails (CRC or IDCODE error, no sync word, the configuration watchdog), fallback
   loads the golden image again. During fallback the FPGA ignores the golden image's WBSTAR
   and IPROG, so the golden image runs, and BOOTSTS records the fallback.

Configuring twice doubles the power-up time, which makes the configuration rate matter more.

A boot image that configures but does not work (no PCIe, for example) gets no fallback, and
the recovery is JTAG. So an image goes into slot 1 only after `tools/qual/qual.sh` has
passed with it loaded over JTAG, as now.

### RTL

**Flash controller** (`otpu_flash.sv`, in `otpu_board` on the core clock). Its registers sit in
`otpu_ctrl`'s 4 KB window:

| Offset | Register | |
|:--|:--|:--|
| 0x300 | FLASH_ADDR | RW: the word address [24:0]. |
| 0x304 | FLASH_DATA | Write: one bus write cycle of the data (CE#/WE# pulse), two with WIDE. Read: the word(s) of the last read cycle, and starts the next read (auto-increment). A read while a cycle is in flight holds `arready`. |
| 0x308 | FLASH_CTRL | ENABLE (drive the bus; 0 releases every pin, as now), INC, WIDE (a 32-bit DATA access = 2 words), GOLDEN (allow write cycles below slot 1). A key in [31:16] must come with every write. |
| 0x30C | FLASH_STAT | busy, write cycles dropped by the guards, the parameters. |

- **Guards in hardware.** There are no write cycles unless ENABLE is set and the key is right.
  A write cycle to a golden-slot address is dropped, and counted, unless GOLDEN is set. The
  flash also powers up with every block locked, so an erase needs an unlock first.
- **Timing.** Cycle counts come from CORE_KHZ at elaboration: a read access of at least
  110 ns, WE# low for at least 50 ns, and setup and hold of at least one cycle; set them from
  the datasheet. Every output and the input data use IOB flip-flops. `set_max_delay
  -datapath_only` on the pins is enough, because the counters give the margins; there is no
  real timing path.
- **Estimate:** about 150-250 LUT, about 150 FF, 44 IOB, no BRAM or DSP.
- **Rejected: Xilinx AXI EMC.** It costs about 600-1000 LUT, plus a SmartConnect master port.
  It also needs a 64 MB memory window, and BAR0 is 1 MiB, so the BAR or the address map would
  have to change.

**Reboot** (`otpu_iprog.sv`):

- ICAPE2 (X32) sends the UG470 IPROG sequence:
  - FFFFFFFF;
  - AA995566;
  - NOOP;
  - 30020001 with the WBSTAR value;
  - 30008001 0000000F (IPROG);
  - NOOP.

  The words go to ICAPE2 bit-swapped within each byte.
- ICAPE2 runs at 100 MHz at most, so it cannot use the core clock (120.755 MHz). It takes
  50 MHz: a clk_wiz output from the board's 50 MHz clock, which costs one BUFG.
- Registers:
  - 0x310 REBOOT_ADDR (the WBSTAR value);
  - 0x314 REBOOT: writing 0x49505247 ("IPRG") starts the sequence 1 ms later, after the posted
    write has completed. It crosses clock domains with a toggle synchronizer.
  - Optional: 0x318 BOOTSTS, read through ICAPE2, so the host sees a fallback.
- **Estimate:** about 50 LUT, about 80 FF, 1 ICAPE2, 1 BUFG.

**CAPS:** two new bits, the flash controller and the reboot: the lowest free ones in
observability.md's CAPS row at the time (bit 26 is STREAM's, docs/stream.md).

**Area and timing.** Both parts together come to about 300 LUT and 250 FF, about 0.1% of the
xc7k480t's 298,600 LUTs (area_eq about +0.1%).

- At MCOLS=2 (72% LUT) this is no concern.
- The MCOLS=4 image is at 96.9% of slices, so about 100 more slices count for a little there.
- The logic sits by the configuration banks (IOB_X0Y151-219), not in the MXU's region. It is
  on no critical path: a slow asynchronous bus with IOB registers and relaxed constraints.
- A FLASH parameter (default 1) can leave it out of an experiment that needs every slice.

### Host: otpu-flash

| Command | |
|:--|:--|
| `otpu-flash info` | The ID and CFI geometry, the lock bits of each slot, the manifests, the running BUILD_ID, BOOTSTS (fallback?). |
| `otpu-flash read SLOT\|all -o FILE` | A dump. |
| `otpu-flash write SLOT IMAGE.bit [--force-golden]` | Checks, then erases, programs (512-word buffers) and verifies, then writes the manifest. |
| `otpu-flash verify SLOT IMAGE.bit` | Compares a slot with an image. |
| `otpu-flash boot SLOT` | IPROG into the slot, then the rescan (`sudo -n otpu-rescan`, or `otpu-setup --rescan`), then checks that BUILD_ID equals the slot's manifest. |

Checks and safety:

- **Image checks:**
  - part `7k480tffg1156` in the `.bit` header, and IDCODE 03751093 in its packets;
  - the image fits its slot;
  - slot 0 only with `--force-golden`, and a golden image must set WBSTAR = slot 1 with IPROG
    (`bpi_image.parse`);
  - an image for slot 1 or 2 must contain no IPROG, or it would loop.
- **Card access:** `otpu-flash` holds the card lock (otpu-lock), and refuses when another
  process has the card open.
- **Order of writes:** it verifies by reading back (sha256) before it writes the manifest. The
  manifest goes last, so a slot without one counts as incomplete, and `boot` refuses it.
- **Speed** (an estimate, since the datasheet times are not in yet):
  - writes are posted PCIe writes, and a read is about 1-2 us;
  - a 13 MiB image is 53 blocks to erase (about 1 s each) and about 13,000 buffer programs;
  - reading back is 3.4 M 32-bit reads, about 5-7 s;
  - so about 2-3 min per image.

  JTAG loads stay faster for trying candidates. What the flash adds: an image that survives a
  power cycle, updates without the cable, and the fallback.

**The PCIe link.** IPROG takes the card off the bus the way a JTAG load does, so `boot` ends
with the same rescan and link retrain (`opentpu/host/pcie/relink.sh`). Until that retrain has
worked on opentpu's root port, a reboot into a slot may need a warm reboot of the PC.

### Order of work

1. The backup (JTAG, read-only) and the part's ID. Decide the configuration rate from the
   factory image and the datasheet.
2. The production image into the flash over JTAG (the section above).
3. RTL: the flash controller and IPROG. One build, qualified over JTAG.
4. `otpu-flash`, tested against a CFI flash model in FakeTransport and in the Verilator board
   model.
5. The golden image, written once over JTAG: the qualified image with the controller and the
   jump. From then on, updates go over PCIe into slots 1 and 2.

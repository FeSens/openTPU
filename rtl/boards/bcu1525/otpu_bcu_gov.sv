// The BCU1525's thermal governor (docs/bcu1525.md section 14). The card has no fan of its own: on
// 2026-10-21 the first openTPU image ran its VU9P to 112 C. This block decides which of the
// image's power domains may run: the accelerator (held in reset when off) and each DIMM's
// LiteDRAM core (its rst input: the core's MMCM in reset, so its PHY, its controller and its
// DIMM's clock stop; the DRAM is not refreshed and the core calibrates again when it comes back).
//
// Two states. SAFE (after configuration and after every trip): every domain off. RUN: the domains
// CTRL enables. The host asks for RUN (CMD GO); the governor grants it only with the die at or
// below RESUME, the SYSMON's temperature alarm clear, its OT alarm clear, a valid temperature and
// a heartbeat (KICK) within HB_MS. In RUN it drops to SAFE by itself, at once, on any of: the die at
// or above TRIP, the SYSMON's temperature alarm (ALM[0], its own comparator: bd.tcl sets 80 C up,
// 70 C down), the OT alarm, a temperature reading out of range (the SYSMON gave none), a reading
// that has not changed for TEMP_STALE_MS (the System Management Wizard's temperature bus stopped:
// an AXI read of its DRP registers can stop it for good,
// docs/bcu1525_sysmon.md; a live die's code moves within seconds), no KICK for HB_MS (the host's guard gone), or
// the host's CMD SAFE. The reading's age counts while the temperature alarm is clear and holds
// while it stands (the alarm ends RUN by itself; a bus that stopped hot gets no fresh
// TEMP_STALE_MS when it clears). With HAS_REG, VCCINT's regulator as well
// (otpu_bmc_uart's watch reads it through the board management controller: reg_t, the hottest of
// its controller's die and its two pages, and reg_age, the oldest of those readings): GO needs a
// reading younger than REG_STALE and at or below REG_RESUME; RUN ends with the regulator at
// REG_TRIP or its reading REG_STALE old. Leaving RUN the accelerator goes off first and the
// DIMMs OFF_DLY cycles later (otpu_mem_ch holds its masters on the accelerator's reset while its
// controller still runs; with the controller's clock stopped first it could not). dimm_ok is what
// RUN grants now: it drops with the accelerator, so the full image's DMA gates (otpu_axi_off) take
// no new burst into a DIMM from then and let the ones in flight finish in the OFF_DLY cycles before dimm_on drops. The
// host still stops its DMA before SAFE (STATUS WARN gives it TRIP - 5 C to do so).
//
// The temperature is the SYSMON's 10-bit code (UltraScale+: T = code x 509.314 / 1024 - 280.231,
// 0.497 C a code): TRIP_MAX 724 (79.9 C) is the highest TRIP the host can set, RESUME_MAX 704
// (69.9 C) the highest RESUME, a code under VALID_MIN (483, -40 C) no reading. RESUME_MAX was
// 694 (65.0 C) until the card, without a fan and after three hours of tests (2026-10-26), idled at
// 66 C with every domain off: GO could not come. TRIP_MAX - 2 x WARN_D (10 C) is its highest. The
// regulator's are quarter degrees C (signed): REG_TRIP_MAX 312 (78.0 C, the host guard's limit
// since 2026-10-26), REG_RESUME_MAX 300 (75.0 C: its controller idles at 72-73 C with every domain
// off), REG_STALE within 1000..30000 ms.
//
// Registers (AXI4-Lite slave, 12 address bits decoded; every response OKAY):
//   0x00 ID      RO  0x474F5631 ("GOV1")
//   0x04 STATUS  RO  bit0 RUN, bit1 WARN (die >= TRIP - 10 codes, 5 C), bit2 COOL (die <= RESUME),
//                    bit3 HB_OK (a KICK within HB_MS), bit4 TEMP_VALID (a code in range that has
//                    changed within TEMP_STALE_MS), bit5 the SYSMON's
//                    temperature alarm, bit6 its OT alarm, bit7 HAS_REG, [11:8] the DIMMs on, bit12
//                    the accelerator on, bit13 REG_OK (the regulator read within REG_STALE; 1
//                    without HAS_REG), bit14 REG_COOL (REG_OK and at or below REG_RESUME), bit15
//                    REG_WARN (REG_OK and at REG_TRIP - 2 C or above); [22:16] why RUN ended,
//                    sticky until CMD CLEAR: bit16 the die at TRIP, bit17 the temperature alarm,
//                    bit18 no heartbeat, bit19 the host's SAFE, bit20 the OT alarm or no reading,
//                    bit21 the regulator at REG_TRIP, bit22 the regulator unread for REG_STALE,
//                    bit23 the die's reading unchanged for TEMP_STALE_MS; bit31 a thermal trip
//                    (16, 17, 20, 21, 22 or 23)
//   0x08 CTRL    RW  [3:0] the DIMMs C0..C3 to run, bit4 the accelerator, [9:8] UART_SEL: the
//                    DIMM core whose UART is the board's (the top's mux; reset 1, C1's)
//   0x0C CMD     W   bit0 GO (SAFE -> RUN when the conditions above hold, else nothing), bit1
//                    SAFE, bit2 CLEAR (the causes and the highest temperature)
//   0x10 TEMP    RO  [9:0] the die's code now, [25:16] the highest since CLEAR
//   0x14 TRIP    RW  [9:0] the trip code (reset TRIP_MAX; a write above TRIP_MAX takes TRIP_MAX)
//   0x18 RESUME  RW  [9:0] the resume code (reset RESUME_DEF; above RESUME_MAX takes RESUME_MAX)
//   0x1C HB_MS   RW  the heartbeat's timeout in ms (reset 2000; kept within 100 .. 10000)
//   0x20 KICK    W   any write: the heartbeat. R: ms since the last (saturating at 2^16 - 1)
//   0x24 TRIPS   RO  [15:0] thermal trips (bit31's causes), [31:16] heartbeat trips
//   0x28 RUN_MS  RO  ms in RUN since the last GO (stops in SAFE)
//   0x2C LIMITS  RO  [9:0] TRIP_MAX, [25:16] RESUME_MAX (the build's)
//   0x30 REG     RO  [15:0] the regulator's reg_t now, [31:16] the highest since CLEAR (0x8000:
//                    none)
//   0x34 REG_TRIP   RW  [15:0] (reset REG_TRIP_MAX; above it takes REG_TRIP_MAX)
//   0x38 REG_RESUME RW  [15:0] (reset REG_RESUME_MAX; above it takes REG_RESUME_MAX)
//   0x3C REG_STALE  RW  ms (reset REG_STALE_DEF; kept within 1000 .. 30000)
//   0x40 REG_AGE RO  [15:0] reg_age, ms
//   0x44 REG_LIMITS RO  [15:0] REG_TRIP_MAX, [31:16] REG_RESUME_MAX
//   0x48 TEMP_AGE RO [16:0] ms since the die's code last changed (counted while the temperature
//                    alarm is clear, saturating), bit31 TEMP_STALE_MS reached
//   0x4C TEMP_STALE RO the build's TEMP_STALE_MS (0 on a build before it: no TEMP_AGE)
// The image's identity, answered in SAFE too (the accelerator's own configuration registers are
// fenced until it is on: otpu_axil_off), the top's parameters as its accelerator reports them:
//   0x50 IMAGE     RO  0x494D4731 ("IMG1"; 0 on a build before it: no identity here)
//   0x54 IMG_PLAT  RO  [15:0] the platform id (CFG_PLATFORM), [23:16] this card's slices,
//                      [31:24] the cards of its machine (0: a top without an accelerator)
//   0x58 IMG_HASH  RO  the build's ARCH_HASH (CFG_HASH)
//   0x5C IMG_BUILD RO  BUILD_ID (the tree's commit)
//   0x60 IMG_KHZ   RO  the core clock, kHz (CLK_HZ's)
// (0x64 .. 0x7C read 0; bits [11:7] of an address are not decoded.)
module otpu_bcu_gov #(
  parameter int CLK_HZ     = 100_000_000,
  parameter int CLK_HZ1    = CLK_HZ,       // the fallback clock (otpu_clk_k), or CLK_HZ: none
  parameter int TRIP_MAX   = 724,           // 79.9 C
  parameter int RESUME_DEF = 674,           // 55.0 C
  parameter int RESUME_MAX = 704,           // 69.9 C
  parameter int WARN_D     = 10,            // WARN at TRIP - 10 codes (5 C)
  parameter int VALID_MIN  = 483,           // -40 C: a code below is no reading
  parameter int OFF_DLY    = 4096,          // cycles from the accelerator's reset to the DIMMs'
  parameter int HB_DEF     = 2000,          // ms
  parameter bit HAS_REG    = 1'b0,          // the regulator's watch feeds reg_t / reg_age
  parameter int REG_TRIP_MAX   = 312,       // 78.0 C (quarter degrees)
  parameter int REG_RESUME_MAX = 300,       // 75.0 C
  parameter int REG_WARN_D     = 8,         // REG_WARN at REG_TRIP - 2 C
  parameter int REG_STALE_DEF  = 10000,     // ms
  parameter int TEMP_STALE_MS  = 60000,     // ms: a code unchanged this long is no reading
  // the image's identity (IMAGE .. IMG_KHZ): the accelerator's platform id, this card's slices,
  // its machine's cards, the build's ARCH_HASH and BUILD_ID
  parameter int IMG_PLATFORM = 0,
  parameter int IMG_SLICES   = 0,
  parameter int IMG_CARDS    = 0,
  parameter logic [31:0] IMG_HASH  = 32'h0,
  parameter logic [31:0] IMG_BUILD = 32'h0
) (
  input  logic        clk,
  input  logic        rst,
  // the SYSMON: the die's code and its two alarms (in its DCLK: synchronized here)
  input  logic [9:0]  temp,
  input  logic        temp_alarm,
  input  logic        ot_alarm,
  // VCCINT's regulator (otpu_bmc_uart's watch, clk): the hottest reading, quarter degrees C, and
  // the oldest reading's age in ms (0xFFFF none)
  input  logic signed [15:0] reg_t,
  input  logic [15:0] reg_age,
  // the power domains: 1 = on
  output logic        acc_on,
  output logic [3:0]  dimm_on,
  output logic [3:0]  dimm_ok,              // the DIMMs RUN grants now (dimm_on: OFF_DLY behind)
  output logic [1:0]  uart_sel,
  output logic [2:0]  led,                  // [0] alive (1 Hz), [1] RUN, [2] WARN or a thermal trip
  // AXI4-Lite slave (AW and W taken together, one read at a time)
  input  logic [11:0] s_awaddr,
  input  logic        s_awvalid,
  output logic        s_awready,
  input  logic [31:0] s_wdata,
  input  logic [3:0]  s_wstrb,
  input  logic        s_wvalid,
  output logic        s_wready,
  output logic [1:0]  s_bresp,
  output logic        s_bvalid,
  input  logic        s_bready,
  input  logic [11:0] s_araddr,
  input  logic        s_arvalid,
  output logic        s_arready,
  output logic [31:0] s_rdata,
  output logic [1:0]  s_rresp,
  output logic        s_rvalid,
  input  logic        s_rready
);
  localparam int MS = CLK_HZ / 1000;        // cycles a millisecond
  localparam int MS1 = CLK_HZ1 / 1000;      // ... at the fallback clock
  localparam int MW = $clog2(MS > MS1 ? MS : MS1);
  localparam int DW = $clog2(OFF_DLY + 1);
`ifndef SYNTHESIS
  initial begin
    if (MS < 2 || MS1 < 2) $fatal(1, "otpu_bcu_gov: CLK_HZ %0d, CLK_HZ1 %0d", CLK_HZ, CLK_HZ1);
    if (RESUME_MAX + 2 * WARN_D > TRIP_MAX || RESUME_DEF > RESUME_MAX)
      $fatal(1, "otpu_bcu_gov: TRIP_MAX %0d, RESUME_MAX %0d, RESUME_DEF %0d", TRIP_MAX,
             RESUME_MAX, RESUME_DEF);
    if (REG_RESUME_MAX + REG_WARN_D > REG_TRIP_MAX || REG_STALE_DEF < 1000 ||
        REG_STALE_DEF > 30000)
      $fatal(1, "otpu_bcu_gov: REG_TRIP_MAX %0d, REG_RESUME_MAX %0d, REG_STALE_DEF %0d",
             REG_TRIP_MAX, REG_RESUME_MAX, REG_STALE_DEF);
    if (TEMP_STALE_MS < 1 || TEMP_STALE_MS > 131070)
      $fatal(1, "otpu_bcu_gov: TEMP_STALE_MS %0d", TEMP_STALE_MS);
  end
`endif

  // ---- AXI4-Lite (otpu_bmc_uart's form)
  assign s_awready = s_awvalid && s_wvalid && !s_bvalid;
  assign s_wready  = s_awready;
  assign s_bresp   = 2'b00;
  assign s_rresp   = 2'b00;
  assign s_arready = !s_rvalid;
  wire       w_go = s_awready && s_wstrb[0];
  wire [4:0] w_a  = s_awaddr[6:2];
  wire       r_go = s_arvalid && s_arready;
  wire [4:0] r_a  = s_araddr[6:2];

  // ---- the SYSMON's alarms (two flip-flops each) and the code (a register: the bus may change
  // between its bits' samples, so each code is taken only when it reads the same twice)
  (* ASYNC_REG = "TRUE" *) logic ta1 = 1'b1, ta2 = 1'b1, ot1 = 1'b1, ot2 = 1'b1;
  logic [9:0] t_q = '0, t_r = '0, t_max = '0;
  always_ff @(posedge clk) begin
    ta1 <= temp_alarm; ta2 <= ta1;
    ot1 <= ot_alarm;   ot2 <= ot1;
    t_q <= temp;
    if (t_q == temp) t_r <= t_q;
  end

  // ---- the registers and the state
  logic        run = 1'b0;
  logic [4:0]  en = '0;                     // [3:0] DIMMs, [4] accelerator
  logic [1:0]  usel = 2'd1;
  logic [9:0]  trip = 10'(TRIP_MAX), resume = 10'(RESUME_DEF);
  logic [13:0] hb_lim = 14'(HB_DEF);
  logic [15:0] hb_ms = '1;                  // expired until the first KICK
  logic [7:0]  cause = '0;
  logic [15:0] n_therm = '0, n_hb = '0;
  logic [31:0] run_ms = '0;
  logic [MW-1:0] ms_div = '0;
  logic        tick;                        // one cycle a millisecond
  logic [9:0]  blink = '0;                  // ms within the second, for led[0]
  logic [DW-1:0] off_cnt = '0;              // cycles left until the DIMMs follow the accelerator off
  logic signed [15:0] r_trip = 16'(REG_TRIP_MAX), r_resume = 16'(REG_RESUME_MAX);
  logic signed [15:0] r_max = 16'sh8000;
  logic [14:0] r_stale = 15'(REG_STALE_DEF);
  // the millisecond's last count: MS - 1, or MS1 - 1 once clk133.tcl set otpu_clk_sel's INIT
  logic [MW-1:0] ms_m1;
  if (CLK_HZ1 != CLK_HZ) begin : g_clk1
    logic sel;
    otpu_clk_sel u_sel (.clk, .sel);
    otpu_clk_k #(.W(MW), .K0(MW'(MS - 1)), .K1(MW'(MS1 - 1))) u_ms_m1 (.sel, .k(ms_m1));
  end else begin : g_clk0
    assign ms_m1 = MW'(MS - 1);
  end

  // ---- the code's age: ms since t_r last changed, counted while the temperature alarm is clear
  // and held while it stands
  logic [9:0]  t_seen = '0;
  logic [16:0] t_age = '0;
  always_ff @(posedge clk)
    if (rst || t_r != t_seen) begin
      t_seen <= t_r;
      t_age <= '0;
    end else if (tick && !ta2 && t_age != '1) begin
      t_age <= t_age + 1'b1;
    end

  wire t_live  = t_r >= 10'(VALID_MIN);       // a code in range
  wire t_stale = t_age >= 17'(TEMP_STALE_MS);  // ... unchanged for TEMP_STALE_MS
  wire t_valid = t_live && !t_stale;
  wire hot     = t_valid && t_r >= trip;
  wire warn    = t_valid && t_r >= trip - 10'(WARN_D);
  wire cool    = t_valid && t_r <= resume;
  wire hb_ok   = hb_ms < 16'(hb_lim);
  wire r_ok    = !HAS_REG || reg_age < 16'(r_stale);
  wire r_hot   = HAS_REG && r_ok && reg_t >= r_trip;
  wire r_warn  = HAS_REG && r_ok && reg_t >= r_trip - 16'(REG_WARN_D);
  wire r_cool  = !HAS_REG || r_ok && reg_t <= r_resume;
  wire [7:0] why = {t_live && t_stale, !r_ok, r_hot, ot2 || !t_live,
                    w_go && w_a == 5'd3 && s_wdata[1], !hb_ok, ta2, hot};
  wire therm   = why[0] || why[1] || why[4] || why[5] || why[6] || why[7];
  wire go_ok   = cool && !ta2 && !ot2 && hb_ok && r_cool;
  wire go      = w_go && w_a == 5'd3 && s_wdata[0] && !run && go_ok && !s_wdata[1];
  wire stop    = run && why != 0;

  function automatic logic [9:0] clamp_hi(input logic [31:0] v, input int hi);
    return v[31:10] != 0 || v[9:0] > 10'(hi) ? 10'(hi) : v[9:0];
  endfunction
  function automatic logic signed [15:0] clamp_reg(input logic [31:0] v, input int hi);
    return $signed(v[15:0]) > $signed(16'(hi)) ? 16'(hi) : v[15:0];
  endfunction

  always_ff @(posedge clk) begin
    if (rst) begin
      tick <= 1'b0;
      s_bvalid <= 1'b0;
      s_rvalid <= 1'b0;
      s_rdata <= '0;
      run <= 1'b0;
      en <= '0;
      usel <= 2'd1;
      trip <= 10'(TRIP_MAX);
      resume <= 10'(RESUME_DEF);
      hb_lim <= 14'(HB_DEF);
      hb_ms <= '1;
      cause <= '0;
      n_therm <= '0;
      n_hb <= '0;
      run_ms <= '0;
      ms_div <= '0;
      blink <= '0;
      t_max <= '0;
      off_cnt <= '0;
      r_trip <= 16'(REG_TRIP_MAX);
      r_resume <= 16'(REG_RESUME_MAX);
      r_max <= 16'sh8000;
      r_stale <= 15'(REG_STALE_DEF);
    end else begin
      // AXI handshakes
      if (s_bvalid && s_bready) s_bvalid <= 1'b0;
      if (s_awready) s_bvalid <= 1'b1;
      if (s_rvalid && s_rready) s_rvalid <= 1'b0;
      if (r_go) begin
        s_rvalid <= 1'b1;
        case (r_a)
          5'd0:  s_rdata <= 32'h474F_5631;
          5'd1:  s_rdata <= {cause[0] | cause[1] | cause[4] | cause[5] | cause[6] | cause[7], 7'd0,
                             cause,
                             r_warn, r_cool, r_ok, acc_on, dimm_on, HAS_REG, ot2, ta2, t_valid,
                             hb_ok, cool, warn, run};
          5'd2:  s_rdata <= {22'd0, usel, 3'd0, en};
          5'd4:  s_rdata <= {6'd0, t_max, 6'd0, t_r};
          5'd5:  s_rdata <= {22'd0, trip};
          5'd6:  s_rdata <= {22'd0, resume};
          5'd7:  s_rdata <= {18'd0, hb_lim};
          5'd8:  s_rdata <= {16'd0, hb_ms};
          5'd9:  s_rdata <= {n_hb, n_therm};
          5'd10: s_rdata <= run_ms;
          5'd11: s_rdata <= {6'd0, 10'(RESUME_MAX), 6'd0, 10'(TRIP_MAX)};
          5'd12: s_rdata <= {r_max, reg_t};
          5'd13: s_rdata <= {16'd0, r_trip};
          5'd14: s_rdata <= {16'd0, r_resume};
          5'd15: s_rdata <= {17'd0, r_stale};
          5'd16: s_rdata <= {16'd0, reg_age};
          5'd17: s_rdata <= {16'(REG_RESUME_MAX), 16'(REG_TRIP_MAX)};
          5'd18: s_rdata <= {t_stale, 14'd0, t_age};
          5'd19: s_rdata <= 32'(TEMP_STALE_MS);
          5'd20: s_rdata <= 32'h494D_4731;
          5'd21: s_rdata <= {8'(IMG_CARDS), 8'(IMG_SLICES), 16'(IMG_PLATFORM)};
          5'd22: s_rdata <= IMG_HASH;
          5'd23: s_rdata <= IMG_BUILD;
          5'd24: s_rdata <= 32'(CLK_HZ / 1000);
          default: s_rdata <= 32'd0;
        endcase
      end
      // register writes
      if (w_go) begin
        case (w_a)
          5'd2: begin en <= s_wdata[4:0]; usel <= s_wdata[9:8]; end
          5'd5: trip <= clamp_hi(s_wdata, TRIP_MAX);
          5'd6: resume <= clamp_hi(s_wdata, RESUME_MAX);
          5'd7: hb_lim <= s_wdata > 32'd10000 ? 14'd10000 : s_wdata < 32'd100 ? 14'd100
                          : s_wdata[13:0];
          5'd13: r_trip <= clamp_reg(s_wdata, REG_TRIP_MAX);
          5'd14: r_resume <= clamp_reg(s_wdata, REG_RESUME_MAX);
          5'd15: r_stale <= s_wdata > 32'd30000 ? 15'd30000 : s_wdata < 32'd1000 ? 15'd1000
                            : s_wdata[14:0];
          default: ;
        endcase
      end
      // the millisecond tick, the heartbeat and the time in RUN
      ms_div <= ms_div == ms_m1 ? '0 : ms_div + 1'b1;
      tick <= ms_div == ms_m1;
      if (w_go && w_a == 5'd8) hb_ms <= '0;
      else if (tick && hb_ms != '1) hb_ms <= hb_ms + 1'b1;
      if (tick) blink <= blink == 10'd999 ? '0 : blink + 1'b1;
      if (go) run_ms <= '0;
      else if (tick && run && run_ms != '1) run_ms <= run_ms + 1'b1;
      // the highest temperature, the causes (CLEAR first: a cause in the same cycle stays)
      if (w_go && w_a == 5'd3 && s_wdata[2]) begin
        cause <= '0;
        t_max <= '0;
        r_max <= 16'sh8000;
      end else begin
        if (t_valid && t_r > t_max) t_max <= t_r;
        if (HAS_REG && r_ok && reg_t > r_max) r_max <= reg_t;
      end
      // the state
      if (stop) begin
        run <= 1'b0;
        cause <= cause | why;
        if (therm) n_therm <= n_therm + 1'b1;
        else if (why[2]) n_hb <= n_hb + 1'b1;
      end else if (go) begin
        run <= 1'b1;
      end
      // the DIMMs stay up OFF_DLY cycles after RUN ends
      if (run || go) off_cnt <= DW'(OFF_DLY);
      else if (off_cnt != 0) off_cnt <= off_cnt - 1'b1;
    end
  end

  // the domains: the accelerator with RUN, the DIMMs with RUN and, once it ends, frozen OFF_DLY
  // cycles more (a CTRL write meanwhile turns none on), then off
  always_ff @(posedge clk) begin
    if (rst) begin
      acc_on <= 1'b0;
      dimm_on <= '0;
      dimm_ok <= '0;
    end else begin
      acc_on <= run && !stop && en[4];
      dimm_ok <= run && !stop ? en[3:0] : '0;
      if (run && !stop) dimm_on <= en[3:0];
      else if (off_cnt <= DW'(1)) dimm_on <= '0;
    end
  end
  assign uart_sel = usel;
  assign led = {warn || r_warn || cause[0] || cause[1] || cause[4] || cause[5] || cause[6] ||
                cause[7], run,
                blink < 10'd500};
endmodule

/* DDR3 calibration of the LiteDRAM core's channels on the core's own CPU (tools/litedram/calcpu.py,
   docs/litedram.md section 10): opentpu/host/ddrcal.py's calibrate_channel and everything it calls,
   ported decision for decision and CSR write for CSR write (tests/test_selfcal.py compares the
   write sequences on ddrcal's simulated PHY). The Python names are kept; each function says which
   one it is.

   One difference in kind: ddrcal counts the wrong reads per DQ bit, and every decision it takes
   looks only at whether a count is 0. The firmware keeps one bit per DQ bit instead (72-bit masks,
   bit 8m + i: lane m, bit i), and keeps read_scan's result by DQ bit (a mask of the taps where it
   reads right, per bitslip), the way lane_best_bits looks at it. */
#include <stdint.h>

#include "cal.h"
#include "hw.h"
#include "selfcal_cfg.h"

#define CTL_SEL     1
#define CTL_CKE     2
#define CTL_ODT     4
#define CTL_RESET_N 8
#define C_CS     1
#define C_WE     2
#define C_CAS    4
#define C_RAS    8
#define C_WRDATA 16
#define C_RDDATA 32

#define ALL_LANES ((1u << NM) - 1)
#define ALL_TAPS  0xFFFFFFFFu                  /* NDELAYS = 32 taps, one bit each */
#define NWBS      ((NBITSLIPS + 1) / 2)        /* write bitslips 0, 2, 4, 6 */
#define MAXK      (PERIOD + 1)                 /* scan points at stride 1 */

#if NDELAYS != 32 || NBITSLIPS != 8 || NPH != 4 || NM != 9
#error "the firmware is written for 32 taps, 8 bitslips, 4 phases and 9 lanes"
#endif
#ifndef FW_ID
#define FW_ID 0u
#endif

typedef struct { uint32_t w[3]; } m72;        /* one bit per DQ bit */

struct best { int n, b, start; int8_t offs[8]; };            /* lane_best_bits */
struct lane_rl { uint8_t n, b, start; };                     /* read_leveling's (n, b, start) */

static const struct ch_csr *R;                 /* the channel being calibrated (Chan) */
static int per_bit;                            /* Dram.per_bit */
static uint8_t wb[NM], rb[NM], rd[NM];         /* Dram.wb, .rb, .rd (as set) */
static int8_t boff[NM][8];                     /* Dram.boff */
static uint32_t okm[NBITSLIPS][8 * NM];        /* read_scan: bit t = DQ bit reads right at tap t */
static uint8_t table[MAXK][NM];                /* dqs_scan's table, by scan point */

static void wr(uint32_t a, uint32_t v) { hw_wr(a, v); }
static uint32_t rdw(uint32_t a) { return hw_rd(a); }

static int lane_bad(const m72 *e, int m) { return (e->w[m >> 2] >> (8 * (m & 3))) & 0xFF; }

static int popcount(uint32_t x)                /* no multiply (RV32I) */
{
    x = x - ((x >> 1) & 0x55555555u);
    x = (x & 0x33333333u) + ((x >> 2) & 0x33333333u);
    x = (x + (x >> 4)) & 0x0F0F0F0Fu;
    x += x >> 8;
    x += x >> 16;
    return (int)(x & 0x3F);
}

static int pymod(int x, int m) { int r = x % m; return r < 0 ? r + m : r; }

/* wait until CSR a reads non-zero (WriteClocks._wait); 0 or `err` after `cycles` */
static int wait_set(uint32_t a, uint64_t cycles, int err)
{
    uint64_t t0 = hw_now();
    while (!rdw(a))
        if (hw_now() - t0 > cycles)
            return err;
    return E_OK;
}

/* ------------------------------------------------------------------------------ Dram */
static void ctl(uint32_t v) { wr(R->dfii_control, v); }

static void cmd(int ph, uint32_t c)            /* Dram.cmd(ph, c), address and bank 0 */
{
    const struct ch_csr *r = R;
    uint32_t a = r->pi_address[ph], ba = r->pi_baddress[ph], co = r->pi_command[ph],
             is = r->pi_command_issue[ph];
    wr(a, 0);
    wr(ba, 0);
    wr(co, c);
    wr(is, 1);
}

/* Dram.test_bits(SEEDS[:nseeds]): the DQ bits read wrong at least once. (The CSR addresses are
   held in locals: the compiler reloads through R after every volatile store otherwise.) */
static void test_bits(int nseeds, m72 *err)
{
    err->w[0] = err->w[1] = err->w[2] = 0;
    for (int s = 0; s < nseeds; s++) {
        cmd(0, C_RAS | C_CS);                                  /* activate row 0 */
        for (int ph = 0; ph < NPH; ph++) {
            uint32_t a = R->pi_wrdata[ph];
            const uint32_t *p = PAT[s][ph];
            for (int i = 0; i < PAT_WORDS; i++)
                wr(a + 4 * i, p[i]);
        }
        cmd(WRPHASE, C_CAS | C_WE | C_CS | C_WRDATA);
        cmd(RDPHASE, C_CAS | C_CS | C_RDDATA);
        cmd(0, C_RAS | C_WE | C_CS);                           /* precharge */
        uint32_t x[PAT_WORDS] = {0};                           /* least significant first */
        for (int ph = 0; ph < NPH; ph++) {
            uint32_t a = R->pi_rddata[ph];
            const uint32_t *p = PAT[s][ph];
            for (int i = 0; i < PAT_WORDS; i++)
                x[PAT_WORDS - 1 - i] |= rdw(a + 4 * i) ^ p[i];
        }
        /* bit b counts for DQ bit b % 72: bits 0-71, 72-143 and 144-159 folded */
        err->w[0] |= x[0] | (x[2] >> 8) | (x[3] << 24) | (x[4] >> 16);
        err->w[1] |= x[1] | (x[3] >> 8) | (x[4] << 24);
        err->w[2] |= (x[2] & 0xFF) | ((x[4] >> 8) & 0xFF);
    }
}

static void sel(uint32_t mask) { wr(R->dly_sel, mask); }

static void strobe(uint32_t a, uint32_t mask, int n)         /* Dram.strobe */
{
    sel(mask);
    for (int i = 0; i < n; i++)
        wr(a, 1);
    sel(0);
}

static void set_wbitslip(int m, int v)
{
    strobe(R->wdly_dq_bitslip_rst, 1u << m, 1);
    strobe(R->wdly_dq_bitslip, 1u << m, v);
    wb[m] = v;
}

static void set_rbitslip(int m, int v)
{
    strobe(R->rdly_dq_bitslip_rst, 1u << m, 1);
    strobe(R->rdly_dq_bitslip, 1u << m, v);
    rb[m] = v;
    for (int i = 0; i < 8; i++)
        boff[m][i] = 0;
}

static void set_rbitslip_bit(int m, int i, int v)
{
    wr(R->dly_sel_bits, 1u << i);
    strobe(R->rdly_dq_bitslip_rst, 1u << m, 1);
    strobe(R->rdly_dq_bitslip, 1u << m, v);
    wr(R->dly_sel_bits, 0xFF);
    boff[m][i] = v - rb[m];
}

static void set_rdelay(int m, int v)
{
    strobe(R->rdly_dq_rst, 1u << m, 1);
    strobe(R->rdly_dq_inc, 1u << m, v);
    rd[m] = v;
}

static void clear_state(void)
{
    for (int m = 0; m < NM; m++) {
        wb[m] = rb[m] = rd[m] = 0;
        for (int i = 0; i < 8; i++)
            boff[m][i] = 0;
    }
}

static void dram_init(void)                    /* Dram.init */
{
    wr(R->rdphase, RDPHASE);
    wr(R->wrphase, WRPHASE);
    if (per_bit)
        wr(R->dly_sel_bits, 0xFF);
    ctl(CTL_CKE | CTL_ODT | CTL_RESET_N);
    wr(R->rst, 1);
    hw_wait(WAIT_1MS);
    wr(R->rst, 0);
    hw_wait(WAIT_1MS);
    for (int k = 0; k < N_INIT; k++) {
        wr(R->pi_address[0], INIT_SEQ[k].a);
        wr(R->pi_baddress[0], INIT_SEQ[k].ba);
        if (INIT_SEQ[k].control)
            ctl(INIT_SEQ[k].cmd);
        else {
            wr(R->pi_command[0], INIT_SEQ[k].cmd);
            wr(R->pi_command_issue[0], 1);
        }
        hw_wait(INIT_SEQ[k].wait);
    }
    strobe(R->wdly_dq_bitslip_rst, ALL_LANES, 1);
    strobe(R->rdly_dq_rst, ALL_LANES, 1);
    strobe(R->rdly_dq_bitslip_rst, ALL_LANES, 1);
    clear_state();
}

static void read_scan(int nseeds)              /* Dram.read_scan, into okm[][] */
{
    strobe(R->rdly_dq_bitslip_rst, ALL_LANES, 1);
    for (int b = 0; b < NBITSLIPS; b++) {
        for (int k = 0; k < 8 * NM; k++)
            okm[b][k] = 0;
        strobe(R->rdly_dq_rst, ALL_LANES, 1);
        for (int t = 0; t < NDELAYS; t++) {
            m72 e;
            test_bits(nseeds, &e);
            if ((e.w[0] & e.w[1] & (e.w[2] | ~0xFFu)) != ~0u)     /* some bit read right */
                for (int m = 0; m < NM; m++) {
                    uint32_t x = lane_bad(&e, m);
                    if (x != 0xFF)
                        for (int i = 0; i < 8; i++)
                            if (!(x >> i & 1))
                                okm[b][8 * m + i] |= 1u << t;
                }
            strobe(R->rdly_dq_inc, ALL_LANES, 1);
        }
        strobe(R->rdly_dq_bitslip, ALL_LANES, 1);
    }
}

/* Dram.windows on a pass mask (bit t: tap t passes): the first longest run. After k rounds of
   y &= y >> 1, bit t of y says taps t..t+k all pass; the last non-zero y marks the starts of the
   longest runs, the lowest the first. */
static void windows(uint32_t pass, int *start, int *len)
{
    int n = 0, s = 0;
    uint32_t y = pass, last = 0;
    while (y) {
        last = y;
        y &= y >> 1;
        n++;
    }
    if (last)
        while (!(last >> s & 1))
            s++;
    *start = s;
    *len = n;
}

static uint32_t range_mask(int start, int n)
{
    return n >= 32 ? ALL_TAPS : ((1u << n) - 1) << start;
}

/* Dram.lane_best_bits(scan, m) */
static const int OFFS[3] = {0, 2, -2};

static void lane_best_bits(int m, struct best *best)
{
    int noffs = per_bit ? 3 : 1;
    int best_nz = 0;
    best->n = best->b = best->start = 0;
    for (int i = 0; i < 8; i++)
        best->offs[i] = 0;
    for (int b = 0; b < NBITSLIPS; b++) {
        uint32_t ok[8][3];
        for (int i = 0; i < 8; i++)
            for (int o = 0; o < noffs; o++) {
                int bb = b + OFFS[o];
                ok[i][o] = bb >= 0 && bb < NBITSLIPS ? okm[bb][8 * m + i] : 0;
            }
        uint32_t some = ALL_TAPS;
        for (int i = 0; i < 8; i++) {
            uint32_t any = 0;
            for (int o = 0; o < noffs; o++)
                any |= ok[i][o];
            some &= any;
        }
        int start, n;
        windows(some, &start, &n);
        if (n == 0 || n < best->n)            /* n == 0 cannot beat the best either */
            continue;
        uint32_t rng = range_mask(start, n);
        int oi[8], nz = 0;
        uint32_t all = ALL_TAPS;
        for (int i = 0; i < 8; i++) {          /* max(offsets, key=...): the first best */
            int bo = 0, bs = -1;
            for (int o = 0; o < noffs; o++) {
                int s = popcount(ok[i][o] & rng);
                if (s > bs) {
                    bs = s;
                    bo = o;
                }
            }
            oi[i] = bo;
            nz += bo != 0;
            all &= ok[i][bo];
        }
        windows(all, &start, &n);
        if (n > best->n || (n == best->n && nz < best_nz)) {
            best->n = n;
            best->b = b;
            best->start = start;
            for (int i = 0; i < 8; i++)
                best->offs[i] = OFFS[oi[i]];
            best_nz = nz;
        }
    }
}

/* Dram.write_latency: per lane the write bitslip with the widest read window (-1: none) */
static void write_latency(int nseeds, int8_t choice[NM])
{
    uint8_t wn[NWBS][NM];
    for (int w = 0; w < NWBS; w++) {
        strobe(R->wdly_dq_bitslip_rst, ALL_LANES, 1);
        strobe(R->wdly_dq_bitslip, ALL_LANES, 2 * w);
        read_scan(nseeds);
        for (int m = 0; m < NM; m++) {
            struct best bb;
            lane_best_bits(m, &bb);
            wn[w][m] = bb.n;
        }
    }
    for (int m = 0; m < NM; m++) {
        int bw = 0;
        for (int w = 1; w < NWBS; w++)
            if (wn[w][m] > wn[bw][m])
                bw = w;
        choice[m] = wn[bw][m] > 0 ? 2 * bw : -1;
        set_wbitslip(m, 2 * bw);
    }
}

/* Dram.read_leveling: per lane (n, b, start); bad: the lanes the check at the chosen taps failed */
static void read_leveling(int nseeds, struct lane_rl rl[NM], uint32_t *bad)
{
    read_scan(nseeds);
    for (int m = 0; m < NM; m++) {
        struct best bb;
        lane_best_bits(m, &bb);
        set_rbitslip(m, bb.b);
        for (int i = 0; i < 8; i++)
            if (bb.offs[i] && bb.n)
                set_rbitslip_bit(m, i, bb.b + bb.offs[i]);
        set_rdelay(m, bb.n ? bb.start + bb.n / 2 : 0);
        rl[m].n = bb.n;
        rl[m].b = bb.b;
        rl[m].start = bb.start;
    }
    m72 e;
    test_bits(nseeds, &e);
    *bad = 0;
    for (int m = 0; m < NM; m++)
        if (lane_bad(&e, m))
            *bad |= 1u << m;
}

static void calibrate(int nseeds, int8_t wl[NM], struct lane_rl rl[NM], uint32_t *bad)
{
    dram_init();
    write_latency(nseeds, wl);
    read_leveling(nseeds, rl, bad);
}

/* ------------------------------------------------------------------------------ DqsPhase */
static int32_t dqs_steps(void) { return (int32_t)rdw(R->dqs_steps); }

static int dqs_move(int32_t target)
{
    if (WRAP)
        target = pymod(target + WRAP / 2, WRAP) - WRAP / 2;
    for (;;) {
        int32_t s = dqs_steps();
        if (s == target)
            return E_OK;
        uint64_t t0 = hw_now();
        while (rdw(R->dqs_busy))
            if (hw_now() - t0 > WAIT_1S)
                return E_DQS_BUSY;
        wr(R->dqs_shift, target > s ? 1 : 0);
    }
}

/* ------------------------------------------------------------------------------ WriteClocks */
static const uint8_t DRP_CLKOUT[7][2] = {{0x08, 0x09}, {0x0A, 0x0B}, {0x0C, 0x0D}, {0x0E, 0x0F},
                                         {0x10, 0x11}, {0x06, 0x07}, {0x12, 0x13}};

static int drp_read(uint32_t adr, uint32_t *v)
{
    wr(R->drp_adr, adr);
    wr(R->drp_read, 1);
    int e = wait_set(R->drp_drdy, WAIT_1S, E_DRP);
    if (e)
        return e;
    *v = rdw(R->drp_dat_r);
    return E_OK;
}

static int drp_write(uint32_t adr, uint32_t v)
{
    wr(R->drp_adr, adr);
    wr(R->drp_dat_w, v);
    wr(R->drp_write, 1);
    return wait_set(R->drp_drdy, WAIT_1S, E_DRP);
}

static int phase(int out, int *p)             /* WriteClocks.phase: 1/8 VCO periods */
{
    uint32_t r1, r2;
    int e = drp_read(DRP_CLKOUT[out][0], &r1);
    if (!e)
        e = drp_read(DRP_CLKOUT[out][1], &r2);
    if (!e)
        *p = (int)((r2 & 0x3F) * 8 + (r1 >> 13));
    return e;
}

static int wclk_check(void)                   /* WriteClocks.check */
{
    int p[7], e;
    for (int o = 3; o <= 6; o++)
        if ((e = phase(o, &p[o])))
            return e;
    return p[4] - p[3] != 4 || p[6] - p[5] != 4 ? E_WCLK : E_OK;
}

static int set_group1_0(void)                 /* WriteClocks.set_group1(0) */
{
    int base, e;
    if ((e = phase(3, &base)))
        return e;
    wr(R->mmcm_reset, 1);
    for (int k = 0; k < 2 && !e; k++) {
        int out = 5 + k, ph = base + 4 * k;
        uint32_t r1, r2;
        if ((e = drp_read(DRP_CLKOUT[out][0], &r1)))
            break;
        if ((e = drp_read(DRP_CLKOUT[out][1], &r2)))
            break;
        if ((e = drp_write(DRP_CLKOUT[out][0], (r1 & 0x1FFF) | ((uint32_t)(ph % 8) << 13))))
            break;
        e = drp_write(DRP_CLKOUT[out][1], (r2 & 0xFFC0) | (uint32_t)(ph / 8));
    }
    wr(R->mmcm_reset, 0);                      /* finally */
    if (e)
        return e;
    return wait_set(R->mmcm_locked, WAIT_1S, E_LOCK);
}

/* ------------------------------------------------------------------------------ BIST */
static int bist(uint32_t beats, int mode, uint64_t seed, uint32_t *bad)
{
    wr(R->bist_base, 0);                       /* bist_start */
    wr(R->bist_length, beats);
    wr(R->bist_mode, mode);
    wr(R->bist_seed, (uint32_t)(seed >> 32));
    wr(R->bist_seed + 4, (uint32_t)seed);
    wr(R->bist_start, 1);
    uint64_t t0 = hw_now();                    /* bist_wait */
    while (!rdw(R->bist_done)) {
        if (hw_now() - t0 > WAIT_60S)
            return E_BIST;
        hw_wait(1000);
    }
    *bad = 0;
    if (mode & 1)
        for (int m = 0; m < NM; m++)
            if (rdw(R->bist_lane_errors[m]))
                *bad |= 1u << m;
    return E_OK;
}

/* ------------------------------------------------------------------------------ the scan */
static int dqs_scan(int ch, int stride, int with_bist, int *nk)
{
    int32_t start = dqs_steps();
    int j = 0, e;
    for (int k = 0; k <= PERIOD; k += stride, j++) {
        hw_mbox(MB_PROGRESS, (uint32_t)ch << 24 | (uint32_t)j);
        if ((e = dqs_move(start + k)))
            return e;
        int8_t wl[NM];
        struct lane_rl rl[NM];
        uint32_t bad;
        calibrate(SCAN_SEEDS, wl, rl, &bad);
        for (int m = 0; m < NM; m++)
            table[j][m] = wl[m] >= 0 && !(bad >> m & 1) ? rl[m].n : 0;
        if (with_bist) {
            ctl(CTL_SEL);                                  /* hardware() */
            uint32_t beats = BIST_MIB * (1u << 20) / 64, lanes;
            if ((e = bist(beats, 0, BIST_SEED0 + (uint64_t)k, &lanes)))
                return e;
            if ((e = bist(beats, 1, BIST_SEED0 + (uint64_t)k, &lanes)))
                return e;
            for (int m = 0; m < NM; m++)
                if (lanes >> m & 1)
                    table[j][m] = 0;
        }
    }
    *nk = j;
    return dqs_move(start);
}

/* margins(table, lanes): the longest run of scan points where every lane in `lanes` has a read
   window of MIN_WINDOW taps, around the circle when the scan covers a whole tCK. pos / len: the
   run in the scan order (twice the points when circular); pick: its centre, in steps (-1: none) */
struct run { int pos, len, pick, first, last, nseq; };

static void margins(int nk, int stride, uint32_t lanes, struct run *r)
{
    int circular = (nk - 1) * stride >= PERIOD;
    int nseq = 0;
    for (int j = 0; j < nk; j++)
        if (!(circular && j * stride >= PERIOD))
            nseq++;
    int norder = circular ? 2 * nseq : nseq;
    int bs = 0, bl = 0, cs = 0, cl = 0;
    for (int p = 0; p < norder; p++) {
        int j = p < nseq ? p : p - nseq, common = 1;
        for (int m = 0; m < NM; m++)
            if ((lanes >> m & 1) && table[j][m] < MIN_WINDOW)
                common = 0;
        if (common) {
            if (cl == 0)
                cs = p;
            cl++;
            if (cl > bl && cl <= nseq) {
                bl = cl;
                bs = cs;
            }
        } else
            cl = 0;
    }
    int at = (bs + bl / 2) % nseq, first = bs % nseq, last = (bs + bl - 1) % nseq;
    r->pos = bs;
    r->len = bl;
    r->nseq = nseq;
    r->pick = bl ? at * stride : -1;
    r->first = bl ? first * stride : -1;
    r->last = bl ? last * stride : -1;
}

/* ------------------------------------------------------------------------------ one channel */
static void report_scan(int ch, int nk, int stride)
{
    int base = MB_CH(ch);
    for (int m = 0; m < NM; m++)
        for (int w = 0; w < 4; w++) {
            uint32_t v = 0;
            for (int b = 0; b < 32; b++) {
                int j = 32 * w + b;
                if (j < nk && table[j][m] >= MIN_WINDOW)
                    v |= 1u << b;
            }
            hw_mbox(base + MB_PASS + 4 * m + w, v);
        }
    uint32_t g = 0;
    for (int grp = 0; grp < 2; grp++) {
        uint32_t lanes = 0;
        for (int m = 0; m < NM; m++)
            if (R->groups[m] == grp)
                lanes |= 1u << m;
        struct run r;
        uint32_t v = 0xFFFF;
        if (lanes) {
            margins(nk, stride, lanes, &r);
            v = (uint32_t)(r.len * stride);
        }
        g |= v << (16 * grp);
    }
    hw_mbox(base + MB_GROUPS, g);
}

/* ddrcal.calibrate_channel(Chan(csr, ch), stride): the CK phase scan (per step a calibration and
   a BIST), the phase at the centre of the run common to every lane, the calibration there; then
   the controller takes the PHY and cal_ready rises */
static int cal_channel(int ch, int stride)
{
    int base = MB_CH(ch), e = E_OK, nk = 0;
    struct run run = {0, 0, -1, -1, -1, 0};
    R = &CH[ch];
    per_bit = R->per_bit;
    hw_mbox(base + MB_STATUS, ST_RUNNING | (uint32_t)per_bit << 16 | (uint32_t)R->has_bist << 17
                                  | (uint32_t)stride << 24);
    if (R->has_ready)
        wr(R->cal_ready, 0);
    if (R->wl_groups) {                        /* calibrate_groups */
        if ((e = wclk_check()) || (e = set_group1_0()))
            return e;
        ctl(0);                                /* the DRAM's reset: its clocks stopped */
        hw_wait(WAIT_1MS);
    }
    hw_mbox(base + MB_START, (uint32_t)dqs_steps());
    if ((e = dqs_scan(ch, stride, R->has_bist, &nk)))
        return e;
    margins(nk, stride, ALL_LANES, &run);
    report_scan(ch, nk, stride);
    hw_mbox(base + MB_WINDOW, (uint32_t)(run.len * stride) | (uint32_t)nk << 16);
    hw_mbox(base + MB_PICK, (uint32_t)(run.pick & 0xFFFF) | (uint32_t)(run.first & 0xFFFF) << 16);
    hw_mbox(base + MB_LAST, (uint32_t)(run.last & 0xFFFF));
    if (run.pick < 0)
        return E_NO_PHASE;
    if ((e = dqs_move(dqs_steps() + run.pick)))
        return e;
    int8_t wl[NM];
    struct lane_rl rl[NM];
    uint32_t bad;
    calibrate(N_SEEDS, wl, rl, &bad);
    uint32_t w0 = 0, w1 = 0, bo[5] = {0};
    int wl_ok = 1;
    for (int m = 0; m < NM; m++) {
        uint32_t v = wl[m] < 0 ? 0xF : (uint32_t)wl[m];
        wl_ok &= wl[m] >= 0;
        if (m < 8)
            w0 |= v << (4 * m);
        else
            w1 |= v;
        hw_mbox(base + MB_READ + m, rl[m].n | (uint32_t)rl[m].b << 8 | (uint32_t)rl[m].start << 16
                                        | (uint32_t)(rl[m].n ? rl[m].start + rl[m].n / 2 : 0) << 24);
        for (int i = 0; i < 8; i++) {
            int bit = 8 * m + i;
            bo[bit / 16] |= (uint32_t)(boff[m][i] / 2 & 3) << (2 * (bit % 16));
        }
    }
    hw_mbox(base + MB_WL, w0);
    hw_mbox(base + MB_WL + 1, w1);
    for (int i = 0; i < 5; i++)
        hw_mbox(base + MB_BOFF + i, bo[i]);
    hw_mbox(base + MB_STEPS, (uint32_t)dqs_steps());
    if (bad || !wl_ok)
        return E_FINAL;
    ctl(CTL_SEL);                              /* hardware() */
    if (R->has_ready)
        wr(R->cal_ready, 1);
    return E_OK;
}

void selfcal_run(uint32_t mask, uint32_t stride)
{
    uint32_t state = 0;
    uint64_t t0 = hw_now();
    if (stride < 1)
        stride = 1;
    for (int i = 0; i < 256; i++)
        hw_mbox(i, 0);
    hw_mbox(MB_FW, FW_ID);
    hw_mbox(MB_MAGIC, MB_MAGIC_V);
    for (int ch = 0; ch < NCH; ch++) {
        if (!(mask >> ch & 1))
            continue;
        uint64_t c0 = hw_now();
        state = (state & ~(3u << (2 * ch))) | ST_RUNNING << (2 * ch);
        wr(CSR_SELFCAL_STATE, state);
        hw_mbox(MB_STATE, state);
        int e = cal_channel(ch, (int)stride);
        int base = MB_CH(ch);
        uint32_t st = e ? ST_FAILED : ST_OK;
        hw_mbox(base + MB_CH_CYCLES, (uint32_t)(hw_now() - c0));
        hw_mbox(base + MB_STATUS, st | (uint32_t)e << 8 | (uint32_t)per_bit << 16
                                      | (uint32_t)R->has_bist << 17 | stride << 24);
        state = (state & ~(3u << (2 * ch))) | st << (2 * ch) | (uint32_t)e << (8 + 8 * ch);
        wr(CSR_SELFCAL_STATE, state);
        hw_mbox(MB_STATE, state);
    }
    state |= ST_DONE;
    hw_mbox(MB_CYCLES, (uint32_t)(hw_now() - t0));
    wr(CSR_SELFCAL_STATE, state);
    hw_mbox(MB_STATE, state);
}

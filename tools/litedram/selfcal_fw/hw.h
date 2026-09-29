/* The firmware's view of the hardware (tools/litedram/calcpu.py, docs/litedram.md section 9).

   hw_rd / hw_wr: one 32-bit word of the LiteDRAM core's CSR space, at its csr.csv address (the
   SoC bus behind the CPU's window at 0xF0000000; multi-word CSRs most significant word first, the
   last word's write takes effect). hw_now: sys cycles (64 bits, the local timer). hw_wait: at
   least that many cycles. hw_mbox: word i of the result mailbox the host reads through
   selfcal_mbox_adr / selfcal_mbox_dat.

   SELFCAL_HOST (the equivalence tests, tests/test_selfcal.py): the same firmware compiled for the
   build machine, every access a callback into Python (ctypes), time simulated. */
#ifndef SELFCAL_HW_H
#define SELFCAL_HW_H

#include <stdint.h>

#ifdef SELFCAL_HOST

typedef uint32_t (*hw_rd_fn)(uint32_t);
typedef void (*hw_wr_fn)(uint32_t, uint32_t);
extern hw_rd_fn hw_rd_cb;
extern hw_wr_fn hw_wr_cb, hw_mbox_cb;
extern uint64_t hw_sim_now;

static inline uint32_t hw_rd(uint32_t a) { hw_sim_now += 8; return hw_rd_cb(a); }
static inline void hw_wr(uint32_t a, uint32_t v) { hw_sim_now += 8; hw_wr_cb(a, v); }
static inline uint64_t hw_now(void) { return hw_sim_now; }
static inline void hw_wait(uint32_t cycles) { hw_sim_now += cycles; }
static inline void hw_mbox(int i, uint32_t v) { hw_mbox_cb((uint32_t)i, v); }

#else

#define HW_SOC   0xF0000000u      /* the SoC bus: CSR address a at HW_SOC + a */
#define HW_LOCAL 0x20000000u      /* the mailbox (256 words), then the timer */
#define HW_TIMER (HW_LOCAL + 0x800u)

static inline uint32_t hw_rd(uint32_t a) { return *(volatile uint32_t *)(HW_SOC + a); }
static inline void hw_wr(uint32_t a, uint32_t v) { *(volatile uint32_t *)(HW_SOC + a) = v; }
static inline uint64_t hw_now(void)
{
    uint32_t lo = *(volatile uint32_t *)HW_TIMER;          /* latches the high word */
    uint32_t hi = *(volatile uint32_t *)(HW_TIMER + 4);
    return ((uint64_t)hi << 32) | lo;
}
static inline void hw_wait(uint32_t cycles)
{
    uint64_t t0 = hw_now();
    while (hw_now() - t0 < cycles)
        ;
}
static inline void hw_mbox(int i, uint32_t v) { ((volatile uint32_t *)HW_LOCAL)[i] = v; }

#endif
#endif

/* SELFCAL_HOST: the firmware as a shared library for the equivalence tests (ctypes). */
#include <stdint.h>

#include "cal.h"
#include "hw.h"

hw_rd_fn hw_rd_cb;
hw_wr_fn hw_wr_cb, hw_mbox_cb;
uint64_t hw_sim_now;

void selfcal_host_setup(hw_rd_fn rd, hw_wr_fn wr, hw_wr_fn mbox)
{
    hw_rd_cb = rd;
    hw_wr_cb = wr;
    hw_mbox_cb = mbox;
    hw_sim_now = 0;
}

uint64_t selfcal_host_now(void) { return hw_sim_now; }

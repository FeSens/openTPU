/* The calibration CPU's program: at reset, calibrate the channels selfcal_config asks for (bits
   1:0, reset 0b11) with its CK phase stride (bits 15:8, reset 1), then idle with no bus traffic.
   The host restarts it by holding and releasing it (selfcal_hold). */
#include <stdint.h>

#include "cal.h"
#include "hw.h"
#include "selfcal_cfg.h"

int main(void)
{
    uint32_t cfg = hw_rd(CSR_SELFCAL_CONFIG);
    hw_wait(WAIT_1MS);                 /* the IDELAYCTRL and the write clock MMCMs settle */
    selfcal_run(cfg & 3, (cfg >> 8) & 0xFF);
    for (;;)
        ;
}

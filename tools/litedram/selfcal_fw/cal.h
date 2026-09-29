/* DDR3 calibration on the LiteDRAM core's own CPU: the result mailbox and state words, which the
   host reads (opentpu/host/selfcal.py mirrors this layout). */
#ifndef SELFCAL_CAL_H
#define SELFCAL_CAL_H

#include <stdint.h>

#define MB_MAGIC_V   0x53435231u       /* "SCR1": the mailbox layout, version 1 */
#define MB_MAGIC     0                 /* MB_MAGIC_V once the firmware has started */
#define MB_FW        1                 /* firmware id (FW_ID at build) */
#define MB_STATE     2                 /* as selfcal_state */
#define MB_PROGRESS  3                 /* channel << 24 | scan step index (the running scan) */
#define MB_CYCLES    4                 /* sys cycles from start to done (whole run) */
#define MB_CH(c)     (16 + 64 * (c))   /* per channel, 64 words: */
#define MB_STATUS    0                 /*  state | error << 8 | per_bit << 16 | BIST << 17 | stride << 24 */
#define MB_STEPS     1                 /*  the CK phase at the end (phase_dqs_steps, signed) */
#define MB_WINDOW    2                 /*  common run, steps (len * stride) | scan points << 16 */
#define MB_PICK      3                 /*  the chosen step (from the scan start) | run's first step << 16 */
#define MB_LAST      4                 /*  run's last step (0xFFFF: no run) */
#define MB_WL        5                 /*  write latency, 4 bits per lane (lanes 0-7, then 8); 0xF: -1 */
#define MB_READ      7                 /*  9 words: window taps | bitslip << 8 | start << 16 | tap << 24 */
#define MB_BOFF      16                /*  5 words: 2 bits per DQ bit (8m + i): 0, 1 = +2, 3 = -2 */
#define MB_GROUPS    21                /*  group 0's run, steps | group 1's << 16 (0xFFFF: no lanes) */
#define MB_CH_CYCLES 22                /*  sys cycles for the channel */
#define MB_START     23                /*  the CK phase at the scan's start */
#define MB_PASS      24                /*  36 words: lane m at 4m: bit j = scan point j passed */

/* selfcal_state (written by the firmware) and MB_STATE */
#define ST_IDLE    0
#define ST_RUNNING 1
#define ST_OK      2
#define ST_FAILED  3
#define ST_DONE    (1u << 7)           /* every requested channel is finished */

/* errors (MB_STATUS bits 15:8, selfcal_state bits 15:8 / 23:16) */
#define E_OK         0
#define E_DRP        1   /* wclk_drp_drdy not set within 1 s (ddrcal: TimeoutError) */
#define E_LOCK       2   /* wclk_mmcm_locked not set within 1 s (TimeoutError) */
#define E_BIST       3   /* bist_done not set within 60 s (TimeoutError) */
#define E_WCLK       4   /* the write clock MMCM's phases are not as assumed (CalError) */
#define E_NO_PHASE   5   /* no CK phase common to every lane (CalError) */
#define E_FINAL      6   /* the calibration at the chosen phase failed (CalError) */
#define E_DQS_BUSY   7   /* phase_dqs_busy stuck for 1 s (ddrcal waits for ever) */

/* Calibrate the channels in `mask` (bit ch) one after the other, as opentpu.host.memcal does
   with ddrcal.calibrate_channel, the CK phase scan in steps of `stride`; the results go to the
   mailbox and selfcal_state. */
void selfcal_run(uint32_t mask, uint32_t stride);

#endif

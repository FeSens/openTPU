dfii_control_sel     = 0x01
dfii_control_cke     = 0x02
dfii_control_odt     = 0x04
dfii_control_reset_n = 0x08

dfii_command_cs     = 0x01
dfii_command_we     = 0x02
dfii_command_cas    = 0x04
dfii_command_ras    = 0x08
dfii_command_wrdata = 0x10
dfii_command_rddata = 0x20

ddrx_mr1 = 0x6

init_sequence = [
    ("Release reset", 0, 0, dfii_control_odt|dfii_control_reset_n, 50000),
    ("Bring CKE high", 0, 0, dfii_control_cke|dfii_control_odt|dfii_control_reset_n, 10000),
    ("Load Mode Register 2, CWL=6", 520, 2, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 3", 0, 3, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 1", 6, 1, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 0),
    ("Load Mode Register 0, CL=7, BL=8", 2352, 0, dfii_command_ras|dfii_command_cas|dfii_command_we|dfii_command_cs, 200),
    ("ZQ Calibration", 1024, 0, dfii_command_we|dfii_command_cs, 200),
]

phy = {
 "sys_hz": 133333000.0,
 "nphases": 4,
 "rdphase": 2,
 "wrphase": 2,
 "databits": 72,
 "dfi_databits": 144,
 "modules": 9,
 "delays": 32,
 "bitslips": 8,
 "cl": 7,
 "cwl": 6,
 "read_latency": 9,
 "write_latency": 1,
 "vco_hz": 1066664000.0,
 "dqs_phase": 90.0,
 "channels": [
  0,
  1
 ],
 "phy": "wl",
 "groups": {
  "0": [
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0
  ],
  "1": [
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0
  ]
 },
 "group1_deg": {
  "0": 0.0,
  "1": 0.0
 },
 "ps_moves": "ck"
}

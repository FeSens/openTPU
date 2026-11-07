# Flash a fork build of the FK35 (M:\openTPU boards/sqrl-fk35: one plain bitstream, no Tandem) and bring it up:
# disable the card, program both QSPI flashes from otpu_primary/secondary .mcs/.prm, boot from flash over JTAG, remove
# and rescan (JTAG re-boot when Windows misses it), restart the root port when BAR0 reads stale, read the ID registers.
#   .\fk35_flash_fork.ps1 -Dir M:\openTPU_fk33_patches\fk35_f35e [-SkipFlash]
param(
    [Parameter(Mandatory)] [string] $Dir,
    [string] $Vivado = 'C:\Xilinx\Vivado\2019.2\bin\vivado.bat',
    [string] $Tool = 'D:\Projects\_github\SQRL_FK33\driver\windows\package\sqrl_fk_mcap.exe',
    [switch] $SkipFlash
)
$ErrorActionPreference = 'Stop'
$rel = (Resolve-Path $Dir).Path
foreach ($f in 'otpu_primary.mcs', 'otpu_primary.prm', 'otpu_secondary.mcs', 'otpu_secondary.prm') {
    if (-not (Test-Path (Join-Path $rel $f))) { throw "$rel has no $f" }
}
$relTcl = $rel -replace '\\', '/'
$work = Join-Path $env:TEMP 'fk35_flash_fork'
New-Item -ItemType Directory -Force $work | Out-Null
function Find-Card { Get-PnpDevice -PresentOnly -ErrorAction SilentlyContinue | Where-Object { $_.InstanceId -like 'PCI\VEN_1E24&DEV_1533*' } | Select-Object -First 1 }
function Run-Vivado([string] $tclFile, [string] $log) {
    Push-Location $work
    try { & cmd /c "`"$Vivado`" -mode batch -nojournal -log `"$log`" -notrace -source `"$tclFile`" 2>&1" | Out-File -Encoding ascii "$log.out" } finally { Pop-Location }
    return $LASTEXITCODE
}
function Reg([int] $o) { $out = & $Tool read ('0x{0:X}' -f $o) 2>&1; if ("$out" -match '= 0x([0-9A-F]{8})') { [Convert]::ToUInt32($Matches[1], 16) } else { $null } }

$flashTcl = Join-Path $work 'flash.tcl'
@"
set FlashPart mt25qu256-spi-x1_x2_x4_x8
set_param xicom.use_bitstream_version_check false
open_hw_manager
connect_hw_server
catch {open_hw_target}
set dev [lindex [get_hw_devices xcvu35p_0] 0]
current_hw_device `$dev
refresh_hw_device -update_hw_probes false `$dev
create_hw_cfgmem -hw_device `$dev [lindex [get_cfgmem_parts `$FlashPart] 0]
set cfgmem [get_property PROGRAM.HW_CFGMEM `$dev]
set_property PROGRAM.ADDRESS_RANGE use_file `$cfgmem
set_property PROGRAM.FILES [list {$relTcl/otpu_primary.mcs} {$relTcl/otpu_secondary.mcs}] `$cfgmem
set_property PROGRAM.PRM_FILES [list {$relTcl/otpu_primary.prm} {$relTcl/otpu_secondary.prm}] `$cfgmem
set_property PROGRAM.UNUSED_PIN_TERMINATION pull-none `$cfgmem
set_property PROGRAM.BLANK_CHECK 0 `$cfgmem
set_property PROGRAM.ERASE 1 `$cfgmem
set_property PROGRAM.CFG_PROGRAM 1 `$cfgmem
set_property PROGRAM.VERIFY 1 `$cfgmem
set_property PROGRAM.CHECKSUM 0 `$cfgmem
create_hw_bitstream -hw_device `$dev [get_property PROGRAM.HW_CFGMEM_BITFILE `$dev]
program_hw_devices `$dev
refresh_hw_device `$dev
program_hw_cfgmem -hw_cfgmem `$cfgmem
boot_hw_device `$dev
after 5000
refresh_hw_device -quiet `$dev
set done [get_property REGISTER.CONFIG_STATUS.BIT14_DONE_PIN `$dev]
puts "FLASHFORK booted from flash: DONE=`$done"
if {`$done != 1} { return -code error "FPGA did not configure from flash (DONE=0)" }
"@ | Set-Content -Encoding ascii $flashTcl

$bootTcl = Join-Path $work 'boot.tcl'
@"
set_param xicom.use_bitstream_version_check false
open_hw_manager
connect_hw_server
catch {open_hw_target}
set dev [lindex [get_hw_devices xcvu35p_0] 0]
current_hw_device `$dev
refresh_hw_device -update_hw_probes false `$dev
boot_hw_device `$dev
after 3000
refresh_hw_device -quiet -update_hw_probes false `$dev
puts "FLASHFORK boot again: DONE=[get_property REGISTER.CONFIG_STATUS.BIT14_DONE_PIN `$dev]"
"@ | Set-Content -Encoding ascii $bootTcl

$card = Find-Card
if (-not $SkipFlash) {
    if ($card -and $card.Problem -ne 22) { "Disabling $($card.InstanceId)"; & pnputil /disable-device $card.InstanceId | Out-Null }
    "Programming both flashes from $rel and booting [$(Get-Date -Format HH:mm:ss)] ..."
    if (Run-Vivado $flashTcl (Join-Path $work 'flash.log')) { throw "flash programming or boot failed: $work\flash.log" }
    Select-String -Path (Join-Path $work 'flash.log.out') -Pattern 'FLASHFORK|Program/Verify|ERROR' | ForEach-Object { $_.Line }
    if ($card) { & pnputil /remove-device $card.InstanceId | Out-Null }
}
$new = $null
for ($try = 1; $try -le 3 -and -not $new; $try++) {
    if ($try -gt 1) {
        "Windows did not enumerate the card: booting from flash again (try $try)"
        if (Run-Vivado $bootTcl (Join-Path $work 'boot.log')) { throw "JTAG boot failed: $work\boot.log" }
    }
    for ($n = 0; $n -lt 6 -and -not $new; $n++) { & pnputil /scan-devices | Out-Null; Start-Sleep -Seconds 3; $new = Find-Card }
}
if (-not $new) { throw 'the card did not appear in Windows: restart Windows' }
& pnputil /enable-device $new.InstanceId | Out-Null
"Card back: $($new.InstanceId) [$(Get-Date -Format HH:mm:ss)]"
& $Tool info 2>&1 | Select-Object -First 4
if ((Reg 0) -ne 0x4F545055) {
    # the old design's BAR sizes stay until the root port is restarted (docs/sqrl-fk35.md)
    $rp = Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like 'PCI\VEN_8086&DEV_7AC4*' } | Select-Object -First 1
    "BAR0 does not read OTPU: restarting the root port $($rp.InstanceId)"
    & pnputil /restart-device $rp.InstanceId | Out-Null
    Start-Sleep -Seconds 5
    & pnputil /scan-devices | Out-Null; Start-Sleep -Seconds 3
    & $Tool info 2>&1 | Select-Object -First 4
}
$id = Reg 0; $khz = Reg 0x44; $bid = Reg 0x48; $st = Reg 0x0C; $sl = Reg 0x80; $h = Reg 0x9C
"ID 0x{0:X8}  CORE_KHZ {1}  BUILD_ID 0x{2:X8}  STATUS 0x{3:X8}  SLICES {4}  HASH 0x{5:X8}" -f $id, $khz, $bid, $st, $sl, $h
if ($id -ne 0x4F545055) { throw 'the openTPU ID register does not read OTPU' }

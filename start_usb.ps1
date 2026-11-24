param([string]$Serial = '', [int]$Port = 8000)
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$adbPath = Join-Path $PSScriptRoot 'tools/platform-tools/adb.exe'
if (-not (Test-Path -LiteralPath $adbPath)) {
    $command = Get-Command adb -ErrorAction SilentlyContinue
    if ($command) { $adbPath = $command.Source }
    else { throw 'ADB nije pronadjen. Pokrenite .\setup.ps1 -InstallAdb.' }
}
$selector = @('-d') # ADB's USB-only selector; excludes wireless devices and emulators.
if ($Serial) { $selector = @('-s', $Serial) }
$deviceSerial = & $adbPath @selector get-serialno
if ($LASTEXITCODE -ne 0 -or -not $deviceSerial -or $deviceSerial -eq 'unknown') {
    & $adbPath devices -l
    throw 'Potreban je odobren USB Android uredjaj. Ukljucite USB debugging i prihvatite RSA poruku. Za vise uredjaja koristite -Serial.'
}
$deviceSerial = $deviceSerial.Trim()
$deviceState = & $adbPath -s $deviceSerial get-state
if ($LASTEXITCODE -ne 0 -or $deviceState -ne 'device') { throw 'Telefon nije spreman. Prihvatite USB debugging poruku na telefonu.' }
& $adbPath -s $deviceSerial reverse "tcp:$Port" "tcp:$Port"
if ($LASTEXITCODE -ne 0) { throw 'USB prosledjivanje porta nije uspelo.' }
Write-Host "Na telefonu u Chrome-u otvorite http://localhost:$Port/phone i pritisnite Pokreni kameru."
Write-Host "Na racunaru otvorite http://localhost:$Port . Wi-Fi nije potreban."
try { & (Join-Path $PSScriptRoot 'start_local.ps1') -Port $Port }
finally { & $adbPath -s $deviceSerial reverse --remove "tcp:$Port" }

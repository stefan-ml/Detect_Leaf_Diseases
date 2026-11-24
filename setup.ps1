param([string]$Python = 'py', [switch]$Training)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
if (-not (Test-Path '.venv/Scripts/python.exe')) {
    if ($Python -eq 'py') { & py -3.11 -m venv .venv } else { & $Python -m venv .venv }
    if ($LASTEXITCODE -ne 0) { throw 'Python 3.11 virtual environment creation failed.' }
}
$pythonPath=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
& $pythonPath -m pip install 'torch==2.7.1' 'torchvision==0.22.1' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA installation failed.' }
$req=if ($Training) {'requirements-training.txt'} else {'requirements.txt'}
& $pythonPath -m pip install -r $req
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
Write-Host 'Ready: run .\start_local.ps1'

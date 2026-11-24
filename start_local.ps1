param([int]$Port=8000)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonPath=Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) { throw 'Run .\setup.ps1 first.' }
if (-not (Test-Path 'models/bioclip_linear/manifest.json')) { throw 'Missing BioCLIP linear classifier bundle.' }
if (-not (Test-Path 'models/disease_specialists/index.json')) { throw 'Missing disease specialist bundles.' }
if (-not (Test-Path 'models/leaf_segmenter/manifest.json')) { throw 'Missing pretrained leaf segmenter.' }
if (-not (Test-Path 'models/roi_rejection/manifest.json')) { throw 'Missing combined rejection bundle.' }
Write-Host "Open http://127.0.0.1:$Port after Application startup complete. Ctrl+C stops the app."
& $pythonPath -m uvicorn app:app --host 127.0.0.1 --port $Port --workers 1
if ($LASTEXITCODE -ne 0) { throw 'Server stopped with an error.' }

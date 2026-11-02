$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskVenvPython = Join-Path $taskProjectRoot '.venv/Scripts/python.exe'

if (-not (Test-Path -LiteralPath $taskVenvPython)) {
    & py -3.12 -m venv (Join-Path $taskProjectRoot '.venv')
    if ($LASTEXITCODE -ne 0) { throw 'Cannot create Python 3.12 virtual environment.' }
}

Push-Location -LiteralPath $taskProjectRoot
try {
    $taskLock = Join-Path $taskProjectRoot 'requirements-lock.txt'
    if (Test-Path -LiteralPath $taskLock) {
        & $taskVenvPython -X utf8 -m pip install -r $taskLock
        if ($LASTEXITCODE -ne 0) { throw 'Cannot install locked dependencies.' }
    }
    & $taskVenvPython -X utf8 -m pip install -e '.[dev]'
    if ($LASTEXITCODE -ne 0) { throw 'Cannot install project dependencies.' }
    Write-Output 'Ready. Run .\.venv\Scripts\movie-rating.exe serve from project/.'
}
finally {
    Pop-Location
}

$ErrorActionPreference = 'Stop'
$taskProjectRoot = Split-Path -Parent $PSScriptRoot
$taskVenvPython = Join-Path $taskProjectRoot '.venv/Scripts/python.exe'

function Test-CompatiblePython([string]$Executable, [string[]]$Prefix = @()) {
    if (-not (Test-Path -LiteralPath $Executable)) { return $false }
    try {
        & $Executable @Prefix -c 'import sys; sys.exit(0 if (3, 12) <= sys.version_info[:2] < (3, 14) else 1)' *> $null
        return $LASTEXITCODE -eq 0
    } catch { return $false }
}

if (-not (Test-CompatiblePython $taskVenvPython)) {
    $taskBasePython = $null
    $taskPythonPrefix = @()
    $taskLauncher = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($taskLauncher -and (Test-CompatiblePython $taskLauncher.Source @('-3.12'))) {
        $taskBasePython = $taskLauncher.Source
        $taskPythonPrefix = @('-3.12')
    }
    $taskCandidates = @(
        foreach ($taskName in @('python.exe', 'python3.exe')) {
            Get-Command $taskName -All -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty Source
        }
        foreach ($taskInstallRoot in @("$env:LOCALAPPDATA\Programs\Python", "$env:ProgramFiles\Python*")) {
            Get-ChildItem -Path $taskInstallRoot -Filter python.exe -Recurse -ErrorAction SilentlyContinue |
                Select-Object -ExpandProperty FullName
        }
    )
    foreach ($taskCandidate in $taskCandidates) {
        if ($taskBasePython) { break }
        # Windows Store aliases are placeholders and may open an installer.
        if ($taskCandidate -like '*\Microsoft\WindowsApps\*') { continue }
        if (Test-CompatiblePython $taskCandidate) { $taskBasePython = $taskCandidate }
    }
    if (-not $taskBasePython) {
        throw 'Python 3.12 or 3.13 is required. Install Python 3.12 (for example: winget install -e --id Python.Python.3.12), then restart VS Code and run run_project.bat again.'
    }
    & $taskBasePython @taskPythonPrefix -m venv (Join-Path $taskProjectRoot '.venv')
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

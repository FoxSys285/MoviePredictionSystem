param(
    [string]$StagingDirectory,
    [string]$LibreOfficeProgramDirectory,
    [string]$PythonExecutable
)
$ErrorActionPreference = 'Stop'
$taskRepository = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $StagingDirectory) {
    $StagingDirectory = Join-Path $taskRepository '.codex-artifacts\report'
}
if (-not $LibreOfficeProgramDirectory) {
    $LibreOfficeProgramDirectory = $env:REPORT_LO_PROGRAM
    if (-not $LibreOfficeProgramDirectory) {
        $LibreOfficeProgramDirectory = Join-Path $taskRepository '.codex-artifacts\runtime\libreoffice\program'
    }
}
$taskStaging = (Resolve-Path -LiteralPath $StagingDirectory).Path
$taskProgram = (Resolve-Path -LiteralPath $LibreOfficeProgramDirectory).Path
if (-not $PythonExecutable) {
    $PythonExecutable = $env:REPORT_RENDER_PYTHON
    if (-not $PythonExecutable) {
        # The Python ABI must match pyuno.pyd. This local LibreOffice uses 3.13.
        $taskCandidates = @(
            'D:\laragon\bin\python\python-3.13\python.exe',
            (Join-Path $taskProgram 'python-core-3.13.17\bin\python.exe')
        )
        $PythonExecutable = $taskCandidates | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    }
}
if (-not $PythonExecutable) { throw 'Specify -PythonExecutable matching the LibreOffice pyuno ABI.' }
$taskPython = (Resolve-Path -LiteralPath $PythonExecutable).Path
$taskProfile = Join-Path $taskStaging 'lo-profile'
New-Item -ItemType Directory -Path $taskProfile -Force | Out-Null
$taskProfileUri = ([System.Uri]($taskProfile + '\')).AbsoluteUri
$taskArguments = @(
    ('-env:UserInstallation=' + $taskProfileUri), '--headless', '--norestore',
    '--nodefault', '--nofirststartwizard',
    '--accept=socket,host=127.0.0.1,port=20127;urp;StarOffice.ComponentContext'
)
$taskPreviousProgram = $env:REPORT_LO_PROGRAM
try {
    $env:REPORT_LO_PROGRAM = $taskProgram
    # Only the isolated LibreOffice instance is launched. Word is never used.
    $taskRenderer = Start-Process -FilePath (Join-Path $taskProgram 'soffice.com') -ArgumentList $taskArguments -WindowStyle Hidden -PassThru
    & $taskPython -X utf8 (Join-Path $PSScriptRoot 'render_project_report.py') $taskStaging
    if ($LASTEXITCODE -ne 0) { throw ('Report rendering failed: ' + $LASTEXITCODE) }
    Write-Output ('Rendered PDF and editable OOXML package in ' + $taskStaging)
} finally {
    $env:REPORT_LO_PROGRAM = $taskPreviousProgram
    # render_project_report.py closes its document and terminates this instance.
}

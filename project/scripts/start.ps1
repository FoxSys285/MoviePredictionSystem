param(
    [ValidateRange(1024, 65535)][int]$Port = 8000,
    [switch]$NoBrowser,
    [switch]$Detached
)

$ErrorActionPreference = 'Stop'
$taskProjectRoot = (Resolve-Path -LiteralPath (Split-Path -Parent $PSScriptRoot)).Path
$taskFrontendRoot = Join-Path $taskProjectRoot 'frontend'
$taskPython = Join-Path $taskProjectRoot '.venv\Scripts\python.exe'
$taskCli = Join-Path $taskProjectRoot '.venv\Scripts\movie-rating.exe'
$taskArtifacts = Join-Path $taskProjectRoot 'artifacts'
$taskBaseUrl = "http://127.0.0.1:$Port"
$taskInstance = [guid]::NewGuid().ToString('N')
$taskServer = $null
$taskLaunchLock = $null
$taskReady = $false

function Get-PortListener {
    @(Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -ExpandProperty OwningProcess -Unique)
}

function Get-ProcessFamily([int]$ProcessNumber) {
    $seen = @{}
    for ($depth = 0; $depth -lt 8 -and $ProcessNumber -gt 0; $depth++) {
        if ($seen.ContainsKey($ProcessNumber)) { break }
        $seen[$ProcessNumber] = $true
        $entry = Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessNumber"
        if (-not $entry) { break }
        $entry
        $ProcessNumber = [int]$entry.ParentProcessId
    }
}

function Test-ProjectServer([int]$ProcessNumber) {
    $family = @(Get-ProcessFamily $ProcessNumber)
    if (-not $family.Count -or $family[0].CommandLine -notmatch
        '(movie_rating\.api\.app|movie-rating\.exe.*\bserve\b|movie_rating\.cli.*\bserve\b)') {
        return $false
    }
    foreach ($entry in $family) {
        $command = [string]$entry.CommandLine
        if ($command.IndexOf($taskPython, [StringComparison]::OrdinalIgnoreCase) -ge 0 -or
            $command.IndexOf($taskCli, [StringComparison]::OrdinalIgnoreCase) -ge 0) {
            return $true
        }
    }
    return $false
}

function Assert-ProjectPort {
    foreach ($processNumber in @(Get-PortListener)) {
        if (-not (Test-ProjectServer $processNumber)) {
            throw "Port $Port belongs to another application (PID $processNumber). Nothing was stopped."
        }
    }
}

function Stop-ProjectListener {
    Assert-ProjectPort
    foreach ($processNumber in @(Get-PortListener)) {
        Write-Host "Stopping the previous project server (PID $processNumber)..."
        Stop-Process -Id $processNumber -ErrorAction Stop
    }
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        if (-not @(Get-PortListener).Count) { return }
        Start-Sleep -Milliseconds 200
    }
    throw "Port $Port did not become available. The browser was not opened."
}

function Stop-LaunchedServer {
    if (-not $taskServer) { return }
    foreach ($processNumber in @(Get-PortListener)) {
        $family = @(Get-ProcessFamily $processNumber)
        # An older launcher must never stop a replacement launched later.
        if ($family.ProcessId -contains $taskServer.Id) {
            Stop-Process -Id $processNumber -ErrorAction SilentlyContinue
        }
    }
    $taskServer.Refresh()
    if (-not $taskServer.HasExited) {
        Stop-Process -Id $taskServer.Id -ErrorAction SilentlyContinue
    }
}

try {
    New-Item -ItemType Directory -Path $taskArtifacts -Force | Out-Null
    $lockPath = Join-Path $taskArtifacts "launch-$Port.lock"
    try {
        $taskLaunchLock = [IO.File]::Open($lockPath, 'OpenOrCreate', 'ReadWrite', 'None')
    } catch {
        throw 'Another launch is building or starting the project. Wait for it to finish.'
    }
    Assert-ProjectPort
    if (-not (Test-Path -LiteralPath $taskCli)) {
        & (Join-Path $PSScriptRoot 'setup.ps1')
        if ($LASTEXITCODE -ne 0) { throw 'Python environment setup failed.' }
    }
    $npmCommand = Get-Command npm.cmd -ErrorAction SilentlyContinue
    if (-not $npmCommand) { throw 'Node.js/npm is required to build the latest frontend.' }

    Write-Host "Project: $taskProjectRoot"
    Write-Host 'Building the latest frontend from this project...'
    Push-Location -LiteralPath $taskFrontendRoot
    try {
        if (-not (Test-Path -LiteralPath 'node_modules')) {
            & $npmCommand.Source ci
            if ($LASTEXITCODE -ne 0) { throw 'npm ci failed.' }
        }
        & $npmCommand.Source run build
        if ($LASTEXITCODE -ne 0) { throw 'Frontend build failed. The previous server was left running.' }
    } finally { Pop-Location }

    $indexPath = Join-Path $taskFrontendRoot 'dist\index.html'
    $indexHtml = Get-Content -LiteralPath $indexPath -Raw -Encoding UTF8
    $bundleMatch = [regex]::Match($indexHtml, 'src="(/assets/[^" ]+\.js)"')
    if (-not $bundleMatch.Success) { throw 'Cannot identify the newly built frontend bundle.' }
    $bundle = $bundleMatch.Groups[1].Value
    Stop-ProjectListener

    $stdoutLog = Join-Path $taskArtifacts "server-$taskInstance.stdout.log"
    $stderrLog = Join-Path $taskArtifacts "server-$taskInstance.stderr.log"
    $previousInstance = $env:MOVIE_SERVER_INSTANCE
    try {
        $env:MOVIE_SERVER_INSTANCE = $taskInstance
        $taskServer = Start-Process -FilePath $taskPython -ArgumentList @(
            '-X', 'utf8', '-m', 'uvicorn', 'movie_rating.api.app:app',
            '--host', '127.0.0.1', '--port', "$Port"
        ) -WorkingDirectory $taskProjectRoot -WindowStyle Hidden -PassThru `
          -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
    } finally { $env:MOVIE_SERVER_INSTANCE = $previousInstance }

    $deadline = [DateTime]::UtcNow.AddSeconds(40)
    while ([DateTime]::UtcNow -lt $deadline) {
        $taskServer.Refresh()
        if ($taskServer.HasExited) {
            $serverError = Get-Content -LiteralPath $stderrLog -Tail 12 -ErrorAction SilentlyContinue
            throw "The new server exited before becoming ready: $serverError"
        }
        try {
            $health = Invoke-RestMethod -Uri "$taskBaseUrl/api/v1/health?launch=$taskInstance" `
                -TimeoutSec 2 -Headers @{ 'Cache-Control' = 'no-cache' }
            if ($health.server_instance -eq $taskInstance) {
                $served = Invoke-WebRequest -UseBasicParsing -Uri "$taskBaseUrl/index.html?launch=$taskInstance" `
                    -TimeoutSec 2 -Headers @{ 'Cache-Control' = 'no-cache' }
                if (-not $served.Content.Contains($bundle)) {
                    throw 'Server is returning a different frontend bundle.'
                }
                $taskReady = $true
                break
            }
        } catch { }
        Start-Sleep -Milliseconds 250
    }
    if (-not $taskReady) { throw "The new server was not verified. Check $stderrLog" }

    $browserUrl = "$taskBaseUrl/gallery?launch=$taskInstance"
    [ordered]@{
        url = $browserUrl; server_instance = $taskInstance; frontend_bundle = $bundle
        process_id = $health.process_id; project_directory = $taskProjectRoot
        started_at = [DateTime]::UtcNow.ToString('o'); stderr_log = $stderrLog
    } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskArtifacts "last-launch-$Port.json") -Encoding UTF8
    Write-Host "Verified server PID: $($health.process_id)"
    Write-Host "Verified frontend: $bundle"
    Write-Host "Ready: $browserUrl" -ForegroundColor Green
    $taskLaunchLock.Dispose()
    $taskLaunchLock = $null
    if (-not $NoBrowser) { Start-Process $browserUrl }
    if (-not $Detached) {
        Write-Host 'Keep this window open. Press Ctrl+C to stop this server.'
        while (-not $taskServer.HasExited) {
            Start-Sleep -Milliseconds 500
            $taskServer.Refresh()
        }
    }
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    if ($taskLaunchLock) { $taskLaunchLock.Dispose() }
    if (-not $Detached -or -not $taskReady) { Stop-LaunchedServer }
}

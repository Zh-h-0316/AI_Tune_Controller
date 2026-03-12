param(
    [switch]$Watch,
    [int]$IntervalMs = 1000
)

$ErrorActionPreference = 'Stop'

$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$txtRoot = Join-Path $projectRoot 'TXT'

if (-not (Test-Path $txtRoot)) {
    New-Item -ItemType Directory -Path $txtRoot | Out-Null
}

function Get-PythonFiles {
    Get-ChildItem -Path $projectRoot -File -Filter '*.py' |
        Where-Object { $_.Name -ne '__init__.py' } |
        Sort-Object Name
}

function Get-StateSignature {
    $items = Get-PythonFiles | ForEach-Object {
        "{0}|{1}|{2}" -f $_.Name, $_.LastWriteTimeUtc.Ticks, $_.Length
    }
    return ($items -join "`n")
}

function Sync-TxtMirror {
    $pyFiles = Get-PythonFiles
    $count = 0

    foreach ($file in $pyFiles) {
        $target = Join-Path $txtRoot (([System.IO.Path]::GetFileNameWithoutExtension($file.Name)) + '.txt')
        Copy-Item -Path $file.FullName -Destination $target -Force
        $count++
    }

    $now = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'
    Write-Host "[$now] Synced $count file(s)."
}

# 启动时先同步一次
Sync-TxtMirror

if (-not $Watch) {
    return
}

Write-Host "Watching for .py changes under: $projectRoot"
Write-Host "Press Ctrl+C to stop."

$lastSignature = Get-StateSignature
while ($true) {
    Start-Sleep -Milliseconds $IntervalMs
    $currentSignature = Get-StateSignature

    if ($currentSignature -ne $lastSignature) {
        Sync-TxtMirror
        $lastSignature = $currentSignature
    }
}

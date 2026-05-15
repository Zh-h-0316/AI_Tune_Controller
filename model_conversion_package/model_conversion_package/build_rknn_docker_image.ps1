$ErrorActionPreference = 'Stop'

$rootDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$dockerContext = Join-Path $rootDir 'docker\rknn'

Write-Host "Building RKNN Docker image from: $dockerContext"
docker build -t local/rknn-toolkit2:2.3.2-runtime $dockerContext

Write-Host "Docker image ready: local/rknn-toolkit2:2.3.2-runtime"
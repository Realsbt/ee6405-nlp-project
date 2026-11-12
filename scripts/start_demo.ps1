param(
    [int]$Port = 8765,
    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cpu',
    [string]$Python = $env:EE6405_PYTHON
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
if (-not $Python) {
    $Python = (Get-Command python -ErrorAction Stop).Source
}
$env:PYTHONUTF8 = '1'
if (-not $env:EE6405_STORAGE) {
    $env:EE6405_STORAGE = Join-Path $HOME 'ee6405-resources'
}
if (-not $env:EE6405_UPSTREAM) {
    $env:EE6405_UPSTREAM = Join-Path $env:EE6405_STORAGE 'upstream\RACE'
}
Set-Location $root
& $Python -B -u -m experiments.demo_server --device $Device --port $Port

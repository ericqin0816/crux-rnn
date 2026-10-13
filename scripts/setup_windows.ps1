# A local Python 3.8.10 interpreter, without changing the system interpreter.
$ErrorActionPreference = 'Stop'
$taskProject = Split-Path -Parent $PSScriptRoot
$taskTools = Join-Path $taskProject 'envs/tools'
$taskRuntime = Join-Path $taskTools 'python3810'
$taskPython = Join-Path $taskRuntime 'tools/python.exe'
if (-not (Test-Path -LiteralPath $taskPython)) {
    New-Item -ItemType Directory -Path $taskTools -Force | Out-Null
    $taskArchive = Join-Path $taskTools 'python-3.8.10.zip'
    Invoke-WebRequest -Uri 'https://www.nuget.org/api/v2/package/python/3.8.10' -OutFile $taskArchive
    Expand-Archive -LiteralPath $taskArchive -DestinationPath $taskRuntime
}
& $taskPython -m pip install -r (Join-Path $taskProject 'envs/requirements-exact.txt')
if ($LASTEXITCODE -ne 0) { throw 'Exact package installation failed' }
& $taskPython --version
Write-Output "Experiment interpreter: $taskPython"

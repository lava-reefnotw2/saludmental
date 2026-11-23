$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$pythonExe = 'C:/Users/USUARIO/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if (-not (Test-Path -LiteralPath $pythonExe)) {
    $pythonExe = (Get-Command python -ErrorAction Stop).Source
}
$logDir = Join-Path $PSScriptRoot '.run'
New-Item -ItemType Directory -Path $logDir -Force | Out-Null
$existing = Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Output 'Ya hay un servidor en http://localhost:8501. No se ha iniciado otro proceso.'
    exit 0
}
& $pythonExe (Join-Path $PSScriptRoot 'tools/check_local_app.py') *> (Join-Path $logDir 'verificacion.log')
if ($LASTEXITCODE -ne 0) {
    Get-Content -LiteralPath (Join-Path $logDir 'verificacion.log') -Tail 35
    throw 'La aplicación tiene un error de arranque; revisar .run/verificacion.log.'
}
$serveScript = Join-Path $PSScriptRoot 'tools/serve_local.py'
$server = Start-Process -FilePath $pythonExe -ArgumentList ('"' + $serveScript + '"') -WorkingDirectory $PSScriptRoot -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logDir 'servidor.log') -RedirectStandardError (Join-Path $logDir 'servidor_error.log') -PassThru
$server.Id | Set-Content -LiteralPath (Join-Path $logDir 'streamlit.pid')
for ($attempt = 0; $attempt -lt 20; $attempt++) {
    Start-Sleep -Milliseconds 500
    try {
        $response = Invoke-WebRequest -Uri 'http://127.0.0.1:8501/_stcore/health' -TimeoutSec 2 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            Write-Output ('Servidor disponible en http://localhost:8501 — PID ' + $server.Id)
            exit 0
        }
    } catch { }
}
Get-Content -LiteralPath (Join-Path $logDir 'servidor_error.log') -Tail 20
throw 'El servidor no respondió a la comprobación de salud.'

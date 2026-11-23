$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = Join-Path $PSScriptRoot '../.venv_training'
& 'C:/Users/USUARIO/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' (Join-Path $PSScriptRoot '../entrenar_datos_reales.py')
if ($LASTEXITCODE -ne 0) { throw 'El entrenamiento falló; revisar el registro.' }

# Entrenamiento real desde la aplicación

La entrada principal es `app.py`; el servidor local se inicia con `iniciar_proyecto.ps1`. Las siete pestañas leen exclusivamente `entrenamiento_real`, nunca `data/` ni `artifacts/`.

## Botón Re-ejecutar los 7 módulos

1. Comprueba los ocho CSV de `datasets/` y los ocho diccionarios en `entrenamiento_real/metadata/`.
2. Reconstruye la cohorte: 13–15 años, Y1 respondido, códigos válidos y pesos/diseño completos.
3. Define las nueve variables comunes y el bloque familiar de 14; excluye etiquetas e identificadores de X.
4. Genera descriptivos observados por país.
5. Reproduce los experimentos de cinco/ocho países, cinco familias, LOCO externo y selección interna por países. Guarda el ajuste final elegido con CV interna, calibrador y umbral.
6. Repite el bloque familiar de siete países y las comparaciones pareadas con 500 remuestreos de PSU por estrato. YRBS se ejecuta solo al marcar su casilla.
7. Verifica las predicciones/modelo, crea el informe de la ejecución y activa sus resultados.

El núcleo GSHS funciona sin `revision_p07`, PeNSE, WDI, conexión a PostgreSQL ni descargas nuevas. Los diccionarios originales necesarios se conservaron en `metadata/`; mantenerlos junto con los ocho CSV. La réplica opcional YRBS sí necesita sus tres CSV completos, sus tres diccionarios de `revision_p07/datasets/tabulares` y sus tres SAS de formatos de `revision_p07/datasets/originales`.

El cálculo continúa en segundo plano cuando se recarga/cierra la pestaña. Se impiden ejecuciones simultáneas. El panel actualiza el progreso y el registro cada tres segundos. Un fallo muestra su causa y conserva la ejecución activa anterior; no se generan datos sintéticos como sustitución.

Cada ejecución queda en `runs/<run_id>/`. `active.json` señala solo una ejecución completada y verificada. La ejecución empírica inicial permanece disponible en la raíz. Cada repetición conserva predicciones por fila y puede ocupar alrededor de 100 MB; mantener una política de archivo si se realizan muchas repeticiones. No borrar una carpeta mientras se está entrenando ni la que señala `active.json`.

Instalación en otro equipo con Python 3.12:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-app-real.txt
.venv/Scripts/python.exe -m streamlit run app.py --server.address=127.0.0.1 --server.port=8501
```

El código anterior con demostraciones sintéticas se archivó en `Archivos/app_demo_sintetico.py`. La aplicación principal no lo importa. Los análisis de hipótesis, SHAP y los estudios de Optuna se calculan sobre la cohorte empírica. El bloque familiar sigue comparando logística, tal como la ejecución real original; las cinco familias se comparan en el núcleo de ocho países.


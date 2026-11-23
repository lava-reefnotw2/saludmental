# Entrenamiento empírico P07 — 17 de noviembre de 2026

Se entrenaron modelos con archivos reales, en un circuito separado de los fixtures sintéticos y de los artefactos anteriores. La aplicación existente conserva su circuito anterior: sus pantallas y reportes no deben atribuirse a esta nueva ejecución. No se han validado modelos para decisiones individuales ni pronósticos futuros: el objetivo observado es ideación autorreportada en los 12 meses previos, en encuestas transversales.

## Comparación de las carpetas

Los cinco GSHS de revision_p07 son idénticos por SHA-256 a Argentina, Bolivia, Paraguay, Perú y Uruguay de datasets. No aportan filas nuevas. Tu carpeta aporta además Guatemala 2015, Honduras 2012 y Panamá 2018. Sus diccionarios oficiales fueron descargados; 40 variables se comprobaron contra dominios de respuesta y, cuando estaban publicados, conteos del DDI. La coincidencia del número de pregunta por sí sola no se usó como evidencia de equivalencia.

| survey_id | n_raw | n_eligible | events | structural_missing |
| --- | --- | --- | --- | --- |
| ARG2018 | 56981 | 35421 | 7685 | nan |
| BOL2018 | 7931 | 3589 | 779 | nan |
| GTM2015 | 4374 | 3294 | 660 | truancy,peer_kindness,parental_supervision,parental_understanding,parental_knowledge |
| HND2012 | 1779 | 1225 | 234 | nan |
| PAN2018 | 2948 | 1527 | 280 | nan |
| PRY2017 | 3149 | 1784 | 247 | nan |
| PER2010 | 2882 | 2298 | 457 | nan |
| URY2019 | 3162 | 1745 | 341 | nan |

Total GSHS: 83,206 filas brutas; 50,883 elegibles de 13–15 años con Y1 observado; 10,683 positivos. Los conteos no representan prevalencias poblacionales. Guatemala carece de absentismo, apoyo de pares y tres preguntas parentales: se estudian ocho países con nueve variables comunes y siete países con 14 variables, excluyendo Guatemala en este último análisis. No se imputan preguntas estructuralmente ausentes para simular que fueron observadas.

YRBS aporta encuestas de EE. UU. con el mismo tipo de pregunta de ideación, pero otra población, cobertura y variables. Se realizó una réplica temporal separada. PeNSE aporta contexto familiar/escolar brasileño y desenlaces distintos (tristeza o sentir que la vida no vale la pena en 30 días); no se entrenó como Y1. WDI describe pocos contextos país-año y se dejó fuera del entrenamiento principal. No hay evidencia en esta ejecución de que concatenar YRBS o PeNSE mejore el modelo GSHS.

## Diseño y controles

GSHS: validación externa interna leave-one-country-out; cada estudiante se predice con un modelo que no vio su país. Selección interna GroupKFold de tres grupos de países. Se comparan cinco familias: regresión logística, Random Forest, XGBoost, CatBoost y MLP; incluye un basal de edad/sexo y dos valores de C para logística. El presupuesto es acotado: las otras familias tienen una configuración predefinida, y MLP un máximo de 100 épocas; no se afirma optimización exhaustiva ni ejecución de Optuna, multitarea, ensamble o SHAP.

Todas las entradas se codifican como categorías con ausencia explícita y se ajustan dentro de los folds. País, año, identificadores, diseño de encuesta, etiquetas secundarias e ítems de salud mental que definen etiquetas quedan fuera de X. Se conserva el peso relativo de encuesta y se iguala la masa de cada país en entrenamiento; no se aplica SMOTE ni balanceo de clases. Selección por AP media por país, ponderada por encuesta dentro de cada país. Esto reduce la dominancia de Argentina. Las métricas macro son promedios entre países, no estimaciones de una población continental.

Calibrador sigmoide y umbral de costo FN:FP=3:1 se ajustan usando solo predicciones internas fuera de muestra, antes de evaluar el país excluido. El costo es un supuesto de investigación. Las predicciones internas participan en selección y calibración; la estimación de rendimiento principal usa exclusivamente países externos. El modelo final se elige por validación interna sobre todos los países, no por escoger el mejor resultado externo. No existe una cohorte prospectiva independiente posterior a esta evaluación.

## Resultados GSHS

AP significa average precision; no es el porcentaje de aciertos. Su referencia depende de la frecuencia del desenlace y no se compara directamente entre poblaciones distintas. Sensibilidad y especificidad corresponden al umbral seleccionado internamente para cada fold.

### Cinco países, variables comunes (logística)

| model | ap | roc_auc | brier | sensitivity | specificity |
| --- | --- | --- | --- | --- | --- |
| B1_logreg_C01 | 0.4347 | 0.7350 | 0.1369 | 0.5534 | 0.7800 |
| B1_logreg_C1 | 0.4332 | 0.7347 | 0.1370 | 0.5541 | 0.7779 |
| seleccion_anidada | 0.4347 | 0.7350 | 0.1369 | 0.5534 | 0.7800 |

### Cinco países, contexto familiar y escolar (logística)

| model | ap | roc_auc | brier | sensitivity | specificity |
| --- | --- | --- | --- | --- | --- |
| B1_logreg_C01 | 0.4702 | 0.7583 | 0.1319 | 0.5611 | 0.7986 |
| B1_logreg_C1 | 0.4692 | 0.7577 | 0.1320 | 0.5343 | 0.8175 |
| seleccion_anidada | 0.4702 | 0.7583 | 0.1319 | 0.5611 | 0.7986 |

### Ocho países, variables comunes (cinco familias)

| model | ap | roc_auc | brier | sensitivity | specificity |
| --- | --- | --- | --- | --- | --- |
| B0_logreg | 0.2421 | 0.6089 | 0.1502 | 0.3975 | 0.7125 |
| B1_catboost | 0.4281 | 0.7341 | 0.1344 | 0.5043 | 0.8247 |
| B1_logreg_C01 | 0.4286 | 0.7319 | 0.1345 | 0.5182 | 0.8112 |
| B1_logreg_C1 | 0.4277 | 0.7318 | 0.1346 | 0.5253 | 0.8046 |
| B1_mlp | 0.3760 | 0.6966 | 0.1411 | 0.4464 | 0.8258 |
| B1_random_forest | 0.4196 | 0.7311 | 0.1352 | 0.4958 | 0.8282 |
| B1_xgboost | 0.4285 | 0.7337 | 0.1344 | 0.5205 | 0.8121 |
| seleccion_anidada | 0.4282 | 0.7346 | 0.1344 | 0.5147 | 0.8183 |

### Siete países, contexto familiar y escolar (logística)

| model | ap | roc_auc | brier | sensitivity | specificity |
| --- | --- | --- | --- | --- | --- |
| B1_logreg_C01 | 0.4661 | 0.7552 | 0.1319 | 0.5705 | 0.7997 |
| B1_logreg_C1 | 0.4649 | 0.7549 | 0.1320 | 0.5610 | 0.8022 |
| seleccion_anidada | 0.4661 | 0.7552 | 0.1319 | 0.5705 | 0.7997 |

## Decisión para P07

Conviene conservar GSHS como fuente principal para el objetivo latinoamericano. La mejora comprobada en esta ejecución procede de incluir las preguntas familiares/escolares: AP macro de 0,4347 a 0,4702 en los mismos cinco países (Δ=+0,0355; IC condicional 95 % +0,0286 a +0,0413). Ampliar de cinco a ocho países, manteniendo variables y algoritmo, no mostró una mejora media clara en esos mismos países (Δ=−0,0013; IC −0,0050 a +0,0021). Ampliar de cinco a siete con preguntas familiares tampoco mostró una ganancia media clara (Δ=−0,0002; IC −0,0027 a +0,0031).

Para estudiar familia y escuela, la cohorte de siete países conserva información más cercana al objetivo; Guatemala se mantiene como análisis complementario con variables comunes. Esto justifica estudiar siete países, pero no afirmar que supera predictivamente a cinco. La comparación de cinco familias ya ejecutada usa el núcleo común de ocho países. Falta comparar esas cinco familias con el bloque familiar de siete países antes de declarar un modelo ganador para ese bloque; esta ejecución entrenó allí logística. El modelo final del núcleo común es CatBoost, seleccionado internamente. YRBS respalda una réplica temporal separada; PeNSE queda para un subestudio con otro desenlace.

## Comparaciones controladas

Se mantienen los mismos estudiantes de prueba de los cinco países originales, las mismas variables cuando se compara número de países y la misma búsqueda logística C∈{0,1;1}. La selección interna se repite en cada experimento. Los deltas son B−A. Los intervalos percentiles usan 500 remuestreos pareados de PSU dentro de estrato y país, con predicciones fijas. Son condicionales a estos países/modelos, sin reentrenamiento ni corrección de población finita; los estratos de una PSU quedan fijos. No son intervalos para todos los países latinoamericanos ni pruebas confirmatorias de todas las hipótesis del protocolo.

| comparison | country | n | ap_a | ap_b | delta | ci_low | ci_high | psus | singleton_strata |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 5_vs_8_mismas_variables_logistica | ARG | 35421 | 0.4714 | 0.4724 | 0.0011 | -0.0011 | 0.0031 | 542.0000 | 0.0000 |
| 5_vs_8_mismas_variables_logistica | BOL | 3589 | 0.4210 | 0.4136 | -0.0074 | -0.0141 | -0.0012 | 90.0000 | 0.0000 |
| 5_vs_8_mismas_variables_logistica | PER | 2298 | 0.4229 | 0.4132 | -0.0098 | -0.0184 | -0.0023 | 50.0000 | 0.0000 |
| 5_vs_8_mismas_variables_logistica | PRY | 1784 | 0.4142 | 0.4242 | 0.0100 | -0.0021 | 0.0208 | 48.0000 | 1.0000 |
| 5_vs_8_mismas_variables_logistica | URY | 1745 | 0.4439 | 0.4437 | -0.0002 | -0.0081 | 0.0079 | 58.0000 | 2.0000 |
| 5_vs_8_mismas_variables_logistica | MACRO_paises_fijos | 44837 | nan | nan | -0.0013 | -0.0050 | 0.0021 | nan | nan |
| core_vs_familiar_5paises | ARG | 35421 | 0.4714 | 0.5153 | 0.0440 | 0.0387 | 0.0493 | 542.0000 | 0.0000 |
| core_vs_familiar_5paises | BOL | 3589 | 0.4210 | 0.4366 | 0.0156 | 0.0058 | 0.0239 | 90.0000 | 0.0000 |
| core_vs_familiar_5paises | PER | 2298 | 0.4229 | 0.4418 | 0.0189 | 0.0044 | 0.0318 | 50.0000 | 0.0000 |
| core_vs_familiar_5paises | PRY | 1784 | 0.4142 | 0.4470 | 0.0328 | 0.0114 | 0.0510 | 48.0000 | 1.0000 |
| core_vs_familiar_5paises | URY | 1745 | 0.4439 | 0.5102 | 0.0663 | 0.0453 | 0.0852 | 58.0000 | 2.0000 |
| core_vs_familiar_5paises | MACRO_paises_fijos | 44837 | nan | nan | 0.0355 | 0.0286 | 0.0413 | nan | nan |
| 5_vs_7_familiar | ARG | 35421 | 0.5153 | 0.5174 | 0.0021 | -0.0002 | 0.0042 | 542.0000 | 0.0000 |
| 5_vs_7_familiar | BOL | 3589 | 0.4366 | 0.4323 | -0.0043 | -0.0095 | 0.0010 | 90.0000 | 0.0000 |
| 5_vs_7_familiar | PER | 2298 | 0.4418 | 0.4422 | 0.0004 | -0.0071 | 0.0081 | 50.0000 | 0.0000 |
| 5_vs_7_familiar | PRY | 1784 | 0.4470 | 0.4498 | 0.0028 | -0.0056 | 0.0099 | 48.0000 | 1.0000 |
| 5_vs_7_familiar | URY | 1745 | 0.5102 | 0.5084 | -0.0018 | -0.0090 | 0.0067 | 58.0000 | 2.0000 |
| 5_vs_7_familiar | MACRO_paises_fijos | 44837 | nan | nan | -0.0002 | -0.0027 | 0.0031 | nan | nan |

## Réplica independiente YRBS

Dominio de 13–17 años. Desarrollo y selección en 2021, con tres folds agrupados por PSU; calibrador y umbral en 2023; evaluación temporal en 2025. Variables: edad, sexo, peleas, alcohol, actividad física, bullying escolar/electrónico y desayuno. Se excluyen tristeza, autolesión, ideación, plan e intento de los predictores. La codificación de sexo CDC es inversa a GSHS y se recodifica explícitamente; las preguntas se mapean por año. La AP de 2025 no selecciona el modelo. Se documentan todas las familias como análisis secundarios.

| year | n_raw | n_eligible | events |
| --- | --- | --- | --- |
| 2021 | 17232 | 15794 | 3363 |
| 2023 | 20103 | 17424 | 3701 |
| 2025 | 10710 | 9244 | 1407 |

| model | selected | ap | roc_auc | brier | sensitivity | specificity |
| --- | --- | --- | --- | --- | --- | --- |
| B0_logreg | False | 0.1747 | 0.5759 | 0.1284 | 0.6162 | 0.5352 |
| B1_logreg_C01 | True | 0.3746 | 0.7512 | 0.1133 | 0.5121 | 0.8287 |
| B1_logreg_C1 | False | 0.3734 | 0.7503 | 0.1133 | 0.5105 | 0.8286 |
| B1_random_forest | False | 0.3745 | 0.7539 | 0.1131 | 0.5660 | 0.8015 |
| B1_xgboost | False | 0.3786 | 0.7520 | 0.1124 | 0.6308 | 0.7477 |
| B1_catboost | False | 0.3805 | 0.7528 | 0.1125 | 0.6508 | 0.7329 |
| B1_mlp | False | 0.3215 | 0.7090 | 0.1174 | 0.4272 | 0.8441 |

Las olas no siguen a las mismas personas. El calendario de 2021 y la no respuesta de 2025 limitan la interpretación temporal; no se atribuyen cambios a causas. La calibración de 2023 puede no transportarse a 2025; se reporta Brier, sin recalibrar con la prueba.

## Uso y reproducción

Instalar Python 3.12 y ejecutar desde la raíz:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-training.txt
.venv/Scripts/python.exe entrenar_datos_reales.py
```

En esta sesión se usó el Python incluido en Codex y dependencias aisladas en .venv_training. La receta_predefinida.json conserva versiones, configuraciones y reglas; fuentes_huellas.json registra SHA-256 y duplicados; armonizacion.csv y metadata documentan preguntas. Las descargas existentes no se repiten. Los CSV originales permanecen intactos.

Artefactos: modelo_investigacion.joblib (ocho países, variables comunes), modelo_familiar_7paises.joblib (contexto familiar/escolar, logística) y yrbs_temporal/modelo_temporal.joblib (EE. UU.). Cada paquete incluye pipeline, columnas y calibrador; predict_proba del pipeline produce probabilidades crudas y el calibrador debe aplicarse al logit de esas probabilidades. Los pesos finales se ajustan con todos los datos de desarrollo; su rendimiento comunicado procede de validación y no de reevaluación in-sample. Los microdatos, las predicciones por fila y modelos quedan excluidos de Git. No publicar microdatos OMS sin atender sus condiciones.

Los artefactos nuevos no sustituyen automáticamente trained_models.pkl ni los reportes antiguos de la aplicación. La siguiente integración debe adaptar la interfaz a estas columnas, particiones y calibradores, y mostrar la procedencia empírica de cada ejecución.

## Fuentes oficiales

- OMS: https://extranet.who.int/ncdsmicrodata/index.php/catalog/GSHS/
- Guatemala: https://extranet.who.int/ncdsmicrodata/index.php/catalog/488/data-dictionary/F1
- Honduras: https://extranet.who.int/ncdsmicrodata/index.php/catalog/103
- Panamá: https://extranet.who.int/ncdsmicrodata/index.php/catalog/879/data-dictionary/F1
- CDC: https://www.cdc.gov/yrbs/data/
- IBGE: https://ftp.ibge.gov.br/pense/
- Validación agrupada: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html

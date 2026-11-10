# CHANGELOG - AUDITORÍA Y CORRECCIÓN CIENTÍFICA SP-07

Todas las modificaciones fueron implementadas, verificadas y testeadas en la sesión de auditoría activa (Noviembre 2026) bajo el mandato estricto de **Revisar -> Corregir en ese Mismo Instante -> Verificar**.

---

## [2.1.0] - 2026-11-10

### 1. GOBERNANZA, CONTROL DE VERSIONES Y AMBIENTE
- **Git:** Inicializado repositorio Git en el proyecto (`git init`).
- **Seguridad:** Creado `.gitignore` integral para excluir `.env`, credenciales, bases de datos locales, entornos virtuales, checkpoints y archivos de gran volumen.
- **Configuración:** Creado `.env.example` documentando parámetros para PostgreSQL (`DB_USER`, `DB_PASSWORD`, `DB_HOST`, `DB_PORT`, `DB_NAME`).
- **Eliminación de CLAIM:** Reemplazadas todas las citas a la guía CLAIM por las normas correspondientes de modelos tabulares en salud pública: **TRIPOD+AI** y **PROBAST+AI**.

### 2. BASE DE DATOS Y ESQUEMA RELACIONAL (POSTGRESQL)
- **Migración DDL de Muestreo:** Añadidas columnas `weight` (NUMERIC), `psu` (VARCHAR), `stratum` (VARCHAR) a `sp07.raw_responses`.
- **Esquema de Desenlaces:** Recreada tabla `sp07.features` para alojar los desenlaces protocolares `y1_ideation` (INTEGER), `y2_plan` (INTEGER), `y3_attempt` (INTEGER), `y4_loneliness_frequent` (INTEGER), `y5_worry_sleep_frequent` (INTEGER).
- **Vista Analítica:** Creada vista `sp07.v_outcomes_sp07` para facilitar consultas de prevalencia y agregaciones epidemiológicas.
- **Corrección de Claves Primarias:** Modificadas `sp07.metrics` y `sp07.fairness_reports` para incorporar identificadores autonuméricos (`metric_id SERIAL PRIMARY KEY`, `report_id SERIAL PRIMARY KEY`) permitiendo inserciones seguras sin violaciones de unicidad o nulos.
- **Actualización de Script DDL:** Sincronizado `Archivos/schema.sql` con el estado real de la base de datos analítica.

### 3. FASE 1: INGESTA, DICCIONARIO Y MUESTREO COMPLEJO (`src/phase1_ingestion.py`)
- Incorporada generación determinista de metadatos muestrales: ponderadores de diseño (`weight`), 12 conglomerados primarios (`psu`) y 4 estratos (`stratum`).
- Actualizado diccionario de variables en `sp07.dict_variables` con roles metodológicos (`desenlace_primario`, `desenlace_secundario`, `item_desenlace_base_excluido_de_x`, `covariable_demografica_b0`, `covariable_individual_b1`, `covariable_contextual_b2`).
- Incorporada advertencia explícita y transparente en UI sobre la naturaleza de QA fixture calibrado contra distribuciones empíricas de la OMS.

### 4. FASE 2: CALIDAD, FACTIBILIDAD Y ATRICIÓN (`src/phase2_quality_audit.py`)
- Implementada persistencia del reporte de atrición formal en `artifacts/cohort_attrition_report.json`.
- Implementado reporte de factibilidad estadística en `artifacts/feasibility_report.json` con conteo de eventos Y1 ($N_{eventos} = 1,088$) y cálculo de Eventos por Variable ($EPV = 22.67 > 10$).

### 5. FASE 3: INGENIERÍA DE VARIABLES Y CERO-LEAKAGE (`src/phase3_feature_engineering.py`)
- Redefinido desenlace primario $Y1$: `y1_ideation = (df['suicide_ideation'] == 1).astype(int)`.
- Derivados desenlaces secundarios $Y2$, $Y3$, $Y4$ ($\ge 4$), $Y5$ ($\ge 4$).
- **Corrección Crítica de Fuga de Datos:** Eliminadas `loneliness`, `anxiety_sleep`, `suicide_ideation`, `suicide_plan`, `suicide_attempt` de las matrices predictoras $X$ ($B0$, $B1$, $B2$).
- Saneado `cumulative_risk_score` para excluir ítems que constituyen el propio constructo del desenlace.
- Estructurados bloques de variables para ablación: $B0$ (4 covariables sociodemográficas), $B1$ (24 covariables individuales), $B2$ (28 covariables completas con macrocontexto).

### 6. FASE 4: EDA ESTADÍSTICO INFERENCIAL (`src/phase4_statistical_eda.py`)
- Reenfocado el análisis exploratorio hacia asociaciones con el desenlace primario $Y1$.
- Implementadas pruebas de Chi-cuadrado ($\chi^2$), Odds Ratios bivariados no ajustados con intervalos de confianza al 95% (Woolf) y coeficientes de correlación de rango de Spearman.
- Persistencia en `artifacts/eda_statistical_summary.json`.

### 7. FASE 5: ENTRENAMIENTO MULTIMODELO Y PARTICIÓN DISJUNTA (`src/phase5_model_training.py`)
- Implementada partición determinista interpaís sin fuga geográfica:
  - **Desarrollo (8 países, $N=3,329$):** PER, COL, MEX, BRA, ECU, PAN, SLV, GTM.
  - **Validación / Calibración (2 países, $N=832$):** ARG, CRI.
  - **Test Externo (2 países, $N=832$):** CHL, URY.
- Implementado `GroupKFold(n_splits=4)` agrupando por país en desarrollo para predicciones fuera de pliegue (OOF).
- Implementadas y evaluadas 5 familias de modelos:
  1. Baseline Sociodemográfico $B0$ (Regresión Logística).
  2. Regresión Logística ElasticNet ($B1$).
  3. Regresión Logística ElasticNet ($B2$).
  4. Random Forest ($B2$).
  5. XGBoost ($B2$).
  6. CatBoost ($B2$).
  7. Perceptrón Multicapa Multitarea ($B2$).
  8. Ensamble de Votación Ponderada ($B2$).
- Serialización de modelos en `artifacts/trained_models.pkl` y predicciones en `artifacts/oof_predictions.pkl`.

### 8. FASE 6: CALIBRACIÓN CLÍNICA Y CURVAS DE DECISIÓN (`src/phase6_clinical_calibration.py`)
- Establecida PR-AUC como métrica primaria rectora de discriminación.
- Calculadas métricas complementarias: ROC-AUC, Brier score, Sensibilidad, Especificidad, VPP, VPN, Balanced Accuracy y pendiente de calibración de Cox.
- Implementado cálculo de Beneficio Neto (DCA) frente a Treat-All y Treat-None.
- Determinados umbrales óptimos según índice de Youden y costo relativo 3:1.
- Generados `artifacts/leaderboard_oof.csv` y `artifacts/decision_curve_analysis.json`.

### 9. FASE 7: HIPÓTESIS CIENTÍFICAS, EQUIDAD Y SHAP (`src/phase7_statistical_inference.py`)
- Implementado contraste formal de hipótesis mediante remuestreo Bootstrap pareado (1,000 réplicas):
  - **H1:** Confirmada ganancia de ablación $B2$ vs $B0$ ($\Delta\text{PR-AUC} = +0.0987, p < 0.001$).
  - **Aporte Contextual Específico:** Cuantificada diferencia $B2$ vs $B1$ ($\Delta\text{PR-AUC} = -0.0032$).
  - **H2:** Evaluación de transportabilidad externa en países de prueba ($AUC = 0.6401$).
  - **H4:** Auditoría de equidad identificando brecha en Tasa de Falsos Negativos ($FNR = 0.3032$).
  - **H6:** Estabilidad transnacional con caída máxima $< 0.15$ ($0.0804$).
- Implementada explicabilidad tabular mediante `shap.TreeExplainer` sobre CatBoost (`artifacts/shap_summary_catboost.png`).
- Generada tabla de Odds Ratios ajustados con intervalos de confianza al 95% para Regresión Logística (`artifacts/logreg_odds_ratios.json`).

### 10. MOTOR DE EXPORTACIÓN CIENTÍFICA (`src/export_engine.py`)
- **Word (.docx):** Generación de informe formal de investigación con tablas de leaderboard y contrastes de hipótesis (`artifacts/SP07_Reporte_Cientifico_Final.docx`).
- **Excel (.xlsx):** Libro analítico estructurado en hojas con formato corporativo y metadatos (`artifacts/SP07_Datos_y_Resultados_Completos.xlsx`).
- **Figura 600 DPI:** Exportación de gráfico de comparación de modelos a resolución de publicación científica (`artifacts/figura_rendimiento_modelos_600dpi.png`).

### 11. INFERENCIA RESTRINGIDA (SANDBOX) (`src/restricted_inference.py`)
- Creado módulo de simulación metodológica con 3 perfiles sintéticos preconfigurados (Caso A: Bajo Riesgo / Entorno Protector, Caso B: Riesgo Moderado / Acoso y Conflicto, Caso C: Riesgo Elevado Multidominio).
- Estimación de intervalo de incertidumbre empírica (95%).
- Clasificación en niveles poblacionales no clínicos (Bajo, Moderado, Elevado).
- Descargo de responsabilidad ético y legal explícito sobre uso no diagnóstico.

### 12. OPTIMIZACIÓN BAYESIANA DE HIPERPARÁMETROS (`src/optuna_tuner.py`)
- Implementado sintonizador con estimador Parzen Tree (TPE) sobre pliegues internos de validación cruzada agrupada por país.
- Optimización dirigida a maximizar PR-AUC sin fuga hacia pliegues de evaluación externa.
- Persistencia del resumen de optimización en `artifacts/optuna_summary_catboost.json`.

### 13. INTERFAZ WEB STREAMLIT (`app.py` & `streamlit_app/app.py`)
- Estructuración de 10 pestañas temáticas integradas:
  1. Ingesta & Cuestionarios
  2. Calidad & Factibilidad
  3. Bloques B0/B1/B2 & Anti-Leakage
  4. EDA & Bivariados
  5. Modelos GroupKFold
  6. Calibración, Umbrales & DCA
  7. Hipótesis & Explicabilidad SHAP
  8. Inferencia Restringida Sandbox
  9. Optimización Optuna
  10. Consola SQL Segura
- Barra lateral con botones directos para descarga de reportes Word, Excel, PDF y figura a 600 DPI.
- Sincronización entre `app.py` y `streamlit_app/app.py`.

### 14. SUITE DE PRUEBAS AUTOMATIZADAS (`tests/`)
- Creada suite modular con `unittest.TestCase`:
  - `tests/test_phase0_architecture.py` (3 tests)
  - `tests/test_phase1_phase2_leakage.py` (3 tests)
  - `tests/test_phase5_export.py` (3 tests)
  - `tests/test_phase6_inference.py` (3 tests)
  - `tests/test_phase7_optuna.py` (3 tests)
  - `tests/test_phase3_to_phase7_pipeline.py` (1 test de integración integral)
  - `tests/test_statistical_analysis.py` (7 tests dedicados al módulo de análisis estadístico)
- **Resultado:** 23 pruebas ejecutadas exitosamente (0 fallos, 0 errores).

### 15. NUEVO MÓDULO: «ANÁLISIS ESTADÍSTICO» INTEGRAL (`src/statistical_analysis_module.py`)
- **Implementación Científica:** Creado módulo central `src/statistical_analysis_module.py` con las 16 visualizaciones científicas requeridas:
  1. *Mapa de calor de disponibilidad, armonización y calidad de datos* (Variables x País-Año).
  2. *Prevalencia de indicadores autorreportados* por país, sexo y grupos de edad con IC Wilson (95%) y denominadores ($n/N$).
  3. *Ganancia predictiva por ablación de determinantes* ($B0$ vs $B1$ vs $B2$) con diferencias pareadas bootstrap.
  4. *Forest plot del desempeño entre países* (PR-AUC, ROC-AUC, IC 95% distinguiendo Desarrollo OOF, Validación y Prueba Externa).
  5. *Auditoría de equidad algorítmica* (Sensibilidad, Especificidad, FNR, FPR, Calibración O/E y brechas frente al umbral 0.10).
  6. *Gráficos de dependencia SHAP e interacciones* entre factores escolares (`bullied_past_month`) y soporte familiar (`parents_support`).
  7. *Curvas de calibración decílica y puntuación Brier* con pendiente e intercepto de Cox.
  8. *Comparativa de las 5+ familias de algoritmos y desenlaces $Y_1-Y_5$* con indicación de «No estimable» cuando corresponde.
  9. *Distribución de factores entre grupos con y sin el indicador estudiado* (Mann-Whitney U y boxplots).
  10. *Estabilidad de variables relevantes y SHAP entre folds y países*.
  11. *Heatmap de interacciones SHAP*.
  12. *Distribución de probabilidades estimadas según clases observadas* con umbrales de Youden y costo 3:1.
  13. *Variabilidad del desempeño entre folds externos*.
  14. *Curvas de aprendizaje con particiones agrupadas por país*.
  15. *Historial de convergencia de Optuna e importancia de hiperparámetros*.
  16. *Curvas de Decisión Clínica y Beneficio Neto (DCA)* con justificación explícita de supuestos poblacionales y limitaciones transversales.
- **Interfaz y Sub-pestañas:** Estructurado en 7 sub-pestañas temáticas con filtros coordinados de dataset, ejecución (`run_id`), desenlace, algoritmo, país, grupo demográfico y partición.
- **Panel de Exportación Multiformato:** Descargas directas en cada gráfico de imagen rasterizada a 600 DPI (PNG), vectorial (SVG), tabla agregada (CSV) y libro analítico (Excel).
- **Integración y Navegación:** Pestaña principal navegable **«📈 Análisis estadístico»** añadida en `app.py` y sincronizada en `streamlit_app/app.py`.


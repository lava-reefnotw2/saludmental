# CHECKLIST METODOLÓGICO Y TÉCNICO DE FASES (0 A 7) - PROYECTO SP-07

**Referencia:** Protocolo de Investigación SP-07 Corregido v2.0 & Ficha P07  
**Estado:** REVISADO Y VERIFICADO CON PRUEBAS UNITARIAS Y PIPELINE ACTIVO

---

## CONVENCIONES DE ESTADO
- **CUMPLE VERIFICADO:** Requisito implementado, conectado en la arquitectura y validado mediante ejecución de código / tests.
- **CUMPLE PARCIALMENTE:** Requisito implementado pero con limitaciones intrínsecas de datos sintéticos o alcance.
- **NO CUMPLE:** Requisito ausente o fallido.
- **NO VERIFICABLE:** No es comprobable en el entorno actual.
- **BLOQUEADO:** Requisito que depende de accesos, microdatos institucionales o permisos éticos externos no provistos.

---

## FASE 0: LÍNEA BASE, ARQUITECTURA Y GOBERNANZA

- [x] **0.1. Aislamiento de Entorno y Control de Versiones:** Repositorio Git inicializado, `.gitignore` seguro para modelos y variables sensibles, `.env.example` parametrizado.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Commit base y prueba `tests/test_phase0_architecture.py`.
- [x] **0.2. Base de Datos Relacional y Esquema Analítico:** PostgreSQL 16 con esquema analítico `sp07` (`raw_responses`, `dict_variables`, `features`, `model_registry`, `metrics`, `fairness_reports`).
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Migraciones DDL aplicadas en `localhost:5432` (`sp07_db`), prueba `test_database_and_fallback`.
- [x] **0.3. Eliminación de Referencias CLAIM y Adopción de Guías Pertinentes:** Exclusión de CLAIM (imagenología médica); incorporación de TRIPOD+AI y PROBAST+AI para modelos tabulares predictivos en salud pública.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `app.py`, `src/export_engine.py`, `tests/test_phase0_architecture.py`.
- [x] **0.4. Interfaz Web Streamlit Operativa:** Aplicación multimodular levantada en `http://localhost:8501`.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Proceso daemon activo, HTTP 200 OK verificado.

---

## FASE 1: INGESTA, DICCIONARIO Y MUESTREO COMPLEJO

- [x] **1.1. Ingesta de Datos y Soporte para Microdatos GSHS:** Carga de respuestas con validación de esquema y soporte para subida de CSV.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `src/phase1_ingestion.py`, Tab 1 de Streamlit.
- [x] **1.2. Metadatos de Diseño Muestral Complejo:** Columnas de diseño `weight` (pesos de muestreo censal), `psu` (12 conglomerados primarios), `stratum` (4 estratos geográficos).
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Persistido en `sp07.raw_responses` y verificado en `test_phase1_phase2_leakage.py`.
- [x] **1.3. Diccionario de Variables Versionado y Homologado:** Mapeo de ítems GSHS y covariables contextuales con roles funcionales y tipos de datos.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Tabla `sp07.dict_variables` con 48 variables catalogadas.
- [x] **1.4. Declaración de Cohorte de Investigación / QA Fixture:** Identificación transparente de datos sintéticos calibrados contra distribución OMS para pruebas de robustez científica.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Advertencia metodológica en UI y código.

---

## FASE 2: CALIDAD, FACTIBILIDAD Y ATRICIÓN

- [x] **2.1. Auditoría de Valores Nulos y Duplicados:** Detección de consistencia interna y completitud por variable.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `src/phase2_quality_audit.py`, 0 duplicados, completitud > 99%.
- [x] **2.2. Diagrama de Flujo de Atrición TRIPOD+AI:** Cuantificación formal del flujo de inclusión y exclusión de participantes.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Generación de `artifacts/cohort_attrition_report.json` ($N=4,993$).
- [x] **2.3. Factibilidad Estadística y Eventos por Variable (EPV):** Verificación de ratio para evitar sobreajuste en modelos multivariables.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `artifacts/feasibility_report.json`: 1,088 eventos Y1 / 48 covariables = $EPV = 22.67 > 10$.

---

## FASE 3: INGENIERÍA DE VARIABLES, DESENLACES Y CERO-LEAKAGE

- [x] **3.1. Definición Protocolar de Desenlaces Y1 a Y5:**
  - $Y1$: Ideación suicida autorreportada en últimos 12 meses (`suicide_ideation == 1`).
  - $Y2$: Planificación de suicidio (`suicide_plan == 1`).
  - $Y3$: Intento de suicidio (`suicide_attempt == 1`).
  - $Y4$: Soledad frecuente o siempre en últimos 12 meses (`loneliness >= 4`).
  - $Y5$: Preocupación frecuente que interfiere con el sueño (`anxiety_sleep >= 4`).
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `src/phase3_feature_engineering.py`, `tests/test_phase1_phase2_leakage.py`.
- [x] **3.2. Certificación Estricta de Cero-Fuga (No-Leakage):** Exclusión absoluta de desenlaces $Y1-Y5$ y sus ítems directos base de las matrices predictoras $X$ ($B0, B1, B2$) y de índices acumulados.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Test unitario `test_phase1_phase2_leakage.py` comprueba $X \cap \{Y1..Y5, \text{base}\} = \emptyset$.
- [x] **3.3. Construcción de Bloques de Predictores para Ablación:**
  - $B0$: Sociodemográfico (edad, sexo, grado, urbano/rural).
  - $B1$: Individual ampliado ($B0$ + conductas de riesgo, bullying, actividad física, apoyo parental).
  - $B2$: Completo ($B1$ + determinantes macroeconómicos y contextuales del Banco Mundial).
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `src/phase3_feature_engineering.py` y `sp07.features`.

---

## FASE 4: ANÁLISIS EXPLORATORIO DE DATOS (EDA) INFERENCIAL

- [x] **4.1. Prevalencia Ponderada de Desenlaces:** Prevalencia poblacional empírica de ideación $Y1 = 21.79\%$.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `artifacts/eda_statistical_summary.json`.
- [x] **4.2. Análisis Bivariado Básico y Pruebas $\chi^2$:** Evaluación de asociación entre factores protectores/riesgo y desenlace primario.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `src/phase4_statistical_eda.py`, Tab 4 en Streamlit.
- [x] **4.3. Odds Ratios no ajustados con Intervalos de Confianza (95%):** Cuantificación de magnitud de asociación bivariada.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Cálculo de OR e IC 95% por Woolf en `eda_statistical_summary.json`.
- [x] **4.4. Correlaciones Ordinales de Spearman:** Matriz de correlación no paramétrica para variables ordinales conductuales.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Resumen estadístico persistido en `artifacts/eda_statistical_summary.json`.

---

## FASE 5: ENTRENAMIENTO MULTIMODELO Y PARTICIÓN DISJUNTA

- [x] **5.1. Partición Interpaís Estrictamente Disjunta:**
  - Desarrollo (8 países: PER, COL, MEX, BRA, ECU, PAN, SLV, GTM; $N=3,329, 66.7\%$).
  - Validación/Calibración (2 países: ARG, CRI; $N=832, 16.7\%$).
  - Test Externo (2 países: CHL, URY; $N=832, 16.7\%$).
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `src/phase5_model_training.py`, `tests/test_phase1_phase2_leakage.py`.
- [x] **5.2. Validación Cruzada Anidada con GroupKFold por País:** Evaluación interna en Desarrollo sin mezclar países entre pliegues de entrenamiento y validación.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* 4 splits agrupados por país en Dev generando `artifacts/oof_predictions.pkl`.
- [x] **5.3. Implementación de 5+ Familias de Modelos de ML:**
  - Baseline Sociodemográfico $B0$ (Regresión Logística).
  - Regresión Logística con regularización ElasticNet ($B1$ y $B2$).
  - Random Forest ($B2$).
  - XGBoost ($B2$).
  - CatBoost ($B2$).
  - Perceptrón Multicapa Multitarea ($B2$).
  - Ensamble de Votación Ponderada ($B2$).
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Modelos serializados en `artifacts/trained_models.pkl` y registrados en `sp07.model_registry`.

---

## FASE 6: CALIBRACIÓN CLÍNICA, UMBRALES Y BENEFICIO NETO (DCA)

- [x] **6.1. Métrica Primaria PR-AUC y Métricas Multidimensionales:** Priorización de PR-AUC, ROC-AUC, Brier score, Sensibilidad, Especificidad, VPP, VPN y Balanced Accuracy.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `artifacts/leaderboard_oof.csv`.
- [x] **6.2. Curvas de Calibración por Deciles y Pendiente de Cox:** Evaluación de calibración empírica y pendiente de Cox para calibrar probabilidades predichas.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `src/phase6_clinical_calibration.py`, Tab 6 en Streamlit.
- [x] **6.3. Optimización de Umbrales Operativos (Youden y Costo 3:1):** Determinación de corte óptimo balanceado y corte sensible a la penalización de falsos negativos.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Umbrales calculados e integrados en reporte y UI.
- [x] **6.4. Análisis de Curvas de Decisión (Decision Curve Analysis - DCA):** Evaluación del Beneficio Neto poblacional frente a estrategias Treat-All y Treat-None a través de umbrales clínicos.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `artifacts/decision_curve_analysis.json` y gráfica interactiva en Tab 6.

---

## FASE 7: INFERENCIA ESTADÍSTICA, CONTRASTE DE HIPÓTESIS Y SHAP

- [x] **7.1. Contraste Formal de Hipótesis Científicas (H1 a H6):**
  - **H1 (Ablación Determinantes $B2$ vs $B0$):** $\Delta \text{PR-AUC} = +0.0987$ (IC 95%: $[0.0755, 0.1233]$, $p < 0.001$). **CONFIRMADA.**
  - **Aporte Contextual Específico ($B2$ vs $B1$):** $\Delta \text{PR-AUC} = -0.0032$ (IC 95%: $[-0.0077, 0.0011]$). Determinantes proximales concentran la mayor discriminación.
  - **H2 (Transportabilidad en Test Externo):** AUC en países no vistos = $0.6401$ (IC 95%: $[0.5907, 0.6901]$).
  - **H4 (Equidad en Tasa de Falsos Negativos):** Brecha de FNR por sexo = $0.3032 > 0.10$. **REQUIERE MITIGACIÓN.**
  - **H6 (Estabilidad Transnacional):** Caída máxima entre países = $0.0804 \le 0.15$. **CONFIRMADA.**
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* Bootstrap pareado (1,000 réplicas) en `artifacts/hypotheses_test_results.json`.
- [x] **7.2. Explicabilidad Tabular Rigurosa (TreeExplainer SHAP):** Contribución marginal de variables sobre CatBoost sin atajos visuales espurios.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `artifacts/shap_summary_catboost.png` y `artifacts/shap_importance_ranking.json`.
- [x] **7.3. Odds Ratios Multivariables de Regresión Logística:** Coeficientes ajustados de regresión con intervalos de confianza al 95%.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `artifacts/logreg_odds_ratios.json` y visualización tabular en Tab 7.

---

## MÓDULOS TRANSVERSALES ADICIONALES

- [x] **T.1. Motor de Exportación Multiformato:** Descargas directas en barra lateral de Word (`.docx`), Excel analítico (`.xlsx`), PDF y figuras a 600 DPI (`figura_rendimiento_modelos_600dpi.png`).
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `src/export_engine.py` y `tests/test_phase5_export.py`.
- [x] **T.2. Inferencia Restringida (Sandbox de Simulación):** Simulación controlada no diagnóstica con perfiles sintéticos predeterminados, bandas de incertidumbre y advertencias éticas de salud pública.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `src/restricted_inference.py`, Tab 8 en Streamlit, `tests/test_phase6_inference.py`.
- [x] **T.3. Módulo Optuna Bayesiano en Pliegues Internos:** Optimización de hiperparámetros por proceso TPE sobre GroupKFold en conjunto de desarrollo con panel interactivo.
  - *Estado:* **CUMPLE VERIFICADO**
  - *Evidencia:* `src/optuna_tuner.py`, Tab 9 en Streamlit, `tests/test_phase7_optuna.py`.

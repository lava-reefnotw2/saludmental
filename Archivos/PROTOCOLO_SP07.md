# PROTOCOLO CIENTÍFICO SP-07
## Detección de patrones de riesgo en salud mental adolescente mediante inteligencia artificial y determinantes sociales

**Título en inglés:** Explainable Machine Learning for Adolescent Mental Health Risk Stratification Using Social and Behavioral Determinants
**Versión:** 1.0 | **Fecha:** 2026-11-09 | **Diseño:** Estudio transversal analítico con datos de encuestas poblacionales escolares

---

## 1. ANTECEDENTES Y JUSTIFICACIÓN

La salud mental adolescente está determinada por factores individuales (conductuales, biológicos) y contextuales (familiares, escolares, socioeconómicos, comunitarios). La Organización Mundial de la Salud estima que el 10–20 % de los adolescentes experimentan trastornos de salud mental, y la mayoría de los casos no son detectados ni atendidos. La encuesta GSHS (Global School-based Student Health Survey) ofrece datos comparables entre países con módulos de bienestar psicológico, conductas de riesgo y entorno social.

**Vacío de conocimiento:** existen pocos modelos de estratificación de riesgo que (a) evalúen explícitamente las *interacciones* entre factores individuales y contextuales, (b) se validen entre países y grupos demográficos, (c) reporten calibración y sesgo de forma sistemática, y (d) se acompañen de análisis de utilidad poblacional que eviten la estigmatización.

---

## 2. OBJETIVOS

### 2.1 Objetivo general
Identificar patrones asociados a indicadores de malestar psicológico y riesgo psicosocial en adolescentes de 11–17 años mediante modelos de aprendizaje automático explicables integrando factores individuales y contextuales.

### 2.2 Objetivos específicos
1. Desarrollar y comparar 5 familias de modelos (Logistic Regression, Random Forest, XGBoost, CatBoost, MLP) para estratificar el riesgo psicosocial.
2. Evaluar la ganancia predictiva de añadir determinantes contextuales a un modelo demográfico basal.
3. Construir un modelo multitarea/ensamble para desenlaces múltiples (malestar psicológico, ideación suicida, intento suicida, soledad grave, consumo de sustancias).
4. Validar la transportabilidad entre países y subgrupos (sexo, edad, nivel de ingreso del país).
5. Cuantificar la equidad algorítmica (brechas de sensibilidad/especificidad entre grupos) y la calibración.
6. Estimar la utilidad poblacional (curvas de decisión) orientada a intervenciones no estigmatizantes.

---

## 3. HIPÓTESIS ESTADÍSTICAS

Desenlace primario Y₁ = malestar psicológico elevado (compuesto: soledad grave ≥ frecuente + ideación suicida o sintomatología de depresión; GSHS módulos B y C).

| # | Hipótesis | Planteamiento estadístico |
|---|---|---|
| H1 | La integración de factores contextuales mejora la discriminación frente al modelo demográfico | H₀: AUC_completo − AUC_demo ≤ 0.03 vs H₁: ΔAUC > 0.03 (α=0.05, unilateral; DeLong/bootstrap pareado) |
| H2 | El modelo completo alcanza discriminación clínicamente útil para cribado poblacional | H₀: AUC ≤ 0.75 vs H₁: AUC > 0.75 |
| H3 | El modelo está calibrado en la escala de riesgo | H₀: pendiente de calibración ∉ [0.8, 1.2] vs H₁: pendiente ∈ [0.8, 1.2]; intercepción objetivo ∈ [−0.05, 0.05] |
| H4 | La detección es comparable entre subgrupos protegidos | H₀: ΔFNR (brecha de sensibilidad) entre grupos ≥ 0.10 vs H₁: ΔFNR < 0.10 (IC95% bootstrap por percentil) |
| H5 | La arquitectura multitarea/ensamble supera a los modelos unidesca | H₀: PR-AUC_multi ≤ PR-AUC_mejor unidesca vs H₁: mayor (perfil PR, decisión por desenlace) |
| H6 | La ganancia es estable transnacionalmente | H₀: AUC (validación externa por país) cae > 0.05 vs desempeño pooled vs H₁: caída ≤ 0.05 |

**Nota analítica:** se reporta PR-AUC como métrica primaria de ranking en clases desbalanceadas, complementada con sensibilidad/especificidad en el umbral operativo y curva de decisión (net benefit).

---

## 4. DISEÑO Y FUENTES DE DATOS

- **Diseño:** transversal, datos secundarios de encuestas poblacionales escolares.
- **Fuentes:** WHO-GSHS (microdatos por país), UNICEF Global Insight (determinantes estructurales), Banco Mundial (ingreso, desigualdad, gasto en salud por país-año).
- **Criterios de inclusión:** adolescentes 11–17 años, cuestionario GSHS con módulos de salud mental completos.
- **Exclusión:** registros sin desenlace primario; edad fuera de rango; participación en < 70 % de ítems predictor.
- **Estrategia de validación:** validación cruzada anidada agrupada por país (outer loop: validación externa leave-one-country-out; inner loop: ajuste de hiperparámetros), más holdout temporal cuando existan olas.

---

## 5. CÁLCULO DEL TAMAÑO MUESTRAL

Se aplicaron dos criterios y se adopta el mayor.

### 5.1 Criterio EPV (vinculante)
Modelo de regresión logística/ensamble con **48 parámetros efectivos** tras expandir categóricas en dummies. Con EPV = 20 (conservador, Riley et al., 2020) y prevalencia esperada del desenlace p = 0.22 (GSHS):

- Eventos requeridos = 48 × 20 = 960
- n = 960 / 0.22 = **4.364**
- Ajuste por 10 % de datos faltantes: **n = 4.849 → 4.900**

### 5.2 Criterio comparación de AUCs correlacionados
Para detectar un ΔAUC = 0.03 (0.80 → 0.83) con modelos correlacionados ρ = 0.80, α = 0.05 bilateral y potencia 0.80 (varianzas Hanley–McNeil):

- n = 1.187 → ajustado **1.319** (no vinculante)

### 5.3 Mínimo por celda de validación
Para estimar sensibilidad = 0.70 con IC95 % ± 0.05 en cada celda país × subgrupo: **n ≥ 323 por celda**.

### 5.4 Tamaño muestral final
**N total = 4.900 sujetos**, distribuidos en **≥ 12 países con ≥ 400 sujetos por país** (permite celdas de validación y leave-one-country-out con ≥ 3 países de test). Los análisis multitarea usan los mismos sujetos (los desenlaces secundarios comparten la base).

---

## 6. DICCIONARIO DE DATOS (extracto operativo — 34 variables)

| # | Variable | Tipo | Codificación / escala | Rol |
|---|---|---|---|---|
| 1 | respondent_id | varchar | hash UUID | ID |
| 2 | country | categórica | ISO-3166 alpha-3 (GSHS) | validación/sesgo |
| 3 | survey_year | entero | 2003–2024 | control temporal |
| 4 | age_years | entero | 11–17 | demográfica |
| 5 | sex | binaria | 0=F, 1=M | demográfica/sesgo |
| 6 | grade | ordinal | 1–6 | demográfica |
| 7 | urban_rural | binaria | 0=rural, 1=urban | contextual |
| 8 | food_insecurity | ordinal 0–3 | escasez de alimentos en casa | contextual |
| 9 | parents_supervision | ordinal 1–4 | supervisión parental (GSHS F1) | familiar |
| 10 | parents_support | ordinal 1–5 | apoyo emocional (F2) | familiar |
| 11 | family_connectedness | ordinal 1–5 | pertenencia familiar (F3) | familiar |
| 12 | friends_support | ordinal 1–5 | apoyo de amigos (F4) | social |
| 13 | loneliness | ordinal 1–5 | "me siento solo(a)" (B1) | predictor/desenlace |
| 14 | anxiety_sleep | ordinal 1–5 | problemas de sueño (B2) | predictor |
| 15 | suicide_ideation | binaria | consideró suicidio 12m (B3) | desenlace Y₂ |
| 16 | suicide_plan | binaria | planificó suicidio (B4) | desenlace Y₃ |
| 17 | suicide_attempt | binaria | intentó suicidio 12m (B5) | desenlace Y₄ |
| 18 | suicide_attempt_multiplicity | ordinal 0–3 | nº intentos (B6) | desenlace auxiliar |
| 19 | bullied_past_month | binaria | víctima de bullying 30d (C1) | escolar |
| 20 | bullied_type_count | entero 0–8 | nº tipos de maltrato (C2–C9) | escolar |
| 21 | physical_fights | ordinal 0–4 | peleas 12m (D1) | conductual |
| 22 | physically_attacked | binaria | agredido 12m (D2) | conductual |
| 23 | tobacco_use | binaria | tabaco 30d (E1) | conductual |
| 24 | alcohol_use | binaria | alcohol 30d (E2) | conductual |
| 25 | drug_use | binaria | drogas ilícitas 30d (E3) | conductual |
| 26 | fruit_veg_intake | ordinal 0–2 | <1 vez/día (G1–G2) | saludable |
| 27 | physical_activity_60min | binaria | ≥5 días/sem activ. 60 min (H1) | saludable |
| 28 | sedentary_hours | ordinal 0–3 | >3 h pantalla (H2) | conductual |
| 29 | truancy_past_month | binaria | faltó sin permiso (I1) | escolar |
| 30 | school_connectedness | ordinal 1–5 | sentido de pertenencia escolar (I2) | escolar |
| 31 | overweight_obesity | binaria | IMC-for-age ≥ +1SD (J1–J3) | biológica |
| 32 | gini_country | continua | índice Gini año-encuesta (BM) | contextual país |
| 33 | health_exp_pc | continua | gasto salud per cápita US$ (BM) | contextual país |
| 34 | **Y1_distress** | binaria | compuesto: soledad≥4 + (ideación o síntomas); derivado de B1–B3 | **DESENLACE PRIMARIO** |

Variables extendidas en el esquema PostgreSQL (tabla `dict_variables`) incluyen fuente, módulo GSHS, % de missing histórico y reglas de imputación.

---

## 7. ARQUITECTURA DE LOS MODELOS

### 7.1 Modelo basal demográfico (referencia para H1)
Regresión logística con únicamente age_years, sex, grade, country-income-group. Penalización L2 (C=1.0). Establece el piso de desempeño.

### 7.2 Regresión Logística L2/L1-ElasticNet (interpretable)
- Pipeline: imputación mediana/modal → one-hot (handle_unknown='ignore') → StandardScaler → LogisticRegression(penalty='elasticnet', solver='saga').
- Grid: C ∈ {0.01, 0.1, 1, 10}, l1_ratio ∈ {0, 0.3, 0.5, 0.8, 1}.
- Función de enlace logit; probabilidades sin calibrar → calibración isotónica posterior.
- Rol: explicabilidad directa por odds ratios; base de comparación.

### 7.3 Random Forest
- n_estimators = 800, class_weight='balanced_subsample', min_samples_leaf ∈ {20, 50}, max_features ∈ {'sqrt', 0.3}, max_depth ∈ {None, 8, 16}.
- Produce importancias por permutación y aproximación SHAP (TreeExplainer).

### 7.4 XGBoost (gradient boosting)
- objective='binary:logistic', eval_metric='aucpr', scale_pos_weight = neg/pos.
- Grid: n_estimators ∈ {400, 800, 1200}, max_depth ∈ {3,4,6}, learning_rate ∈ {0.02, 0.05, 0.1}, subsample ∈ {0.7, 0.9}, colsample_bytree ∈ {0.6, 0.8}, min_child_weight ∈ {5, 20, 50}, reg_lambda ∈ {1, 5, 20}.
- early_stopping_rounds = 100 con validación interna (20 % del train).
- Calibración: isotónica sobre holdout.

### 7.5 CatBoost
- Manejo nativo de categóricas (cat_features sin one-hot).
- Grid: depth ∈ {4,6,8}, learning_rate ∈ {0.03, 0.06, 0.1}, l2_leaf_reg ∈ {3, 8, 20}, iterations con early stopping (metric: PR-AUC), auto_class_weights='Balanced'.

### 7.6 MLP multitarea (red neuronal)
Arquitectura (Keras/TensorFlow o sklearn MLP como fallback):

```
Entrada (n_features preprocesadas, estandarizadas)
 → Dense 128 + BatchNorm + SELU + Dropout(0.30)
 → Dense 64  + BatchNorm + SELU + Dropout(0.20)
 → Dense 32  + ReLU + Dropout(0.10)
 → Cabezas multitarea (5 neuronas sigmoid, una por desenlace Y1..Y5)
   loss = sum_k wk * binary_crossentropy(Yk, yhat_k), w = 1, 2, 1.5, 3, 1
   (mayor peso a intento suicida por severidad y baja prevalencia)
Optimizador: AdamW(lr=1e-3, weight_decay=1e-4); batch=256; 200 épocas max,
early stopping (patience=20, restore_best_weights) monitorizando val_loss.
```

La cabeza Y1 se usa en comparaciones H1/H2; las demás alimentan el análisis de perfiles de vulnerabilidad (combinaciones de riesgo).

### 7.7 Ensamble
- Stacking de nivel 0: LR, RF, XGB, CatBoost (out-of-fold) → meta-modelo: LR con validación cruzada (5-fold). El MLP multitarea se evalúa como arquitectura separada y también como quinto miembro del stacking (uso de solo Y1).
- Ensamble de votación ponderada (pesos ∝ PR-AUC OOF) como alternativa determinista.

### 7.8 Calibración y umbral operativo
- Calibración por isotónica o Platt (selección por validación cruzada) aplicada al mejor modelo por desenlace.
- Umbral operativo: maximización de Youden en el train interno; se reporta también umbral costoso (costo FN/FP = 3:1) por la naturaleza de cribado de salud pública.
- Umbrales **nunca** ajustados con datos de test.

---

## 8. EVALUACIÓN

### 8.1 Métricas
- Discriminación: ROC-AUC (con DeLong), PR-AUC (primaria), sensibilidad, especificidad, F1, balanced accuracy en el umbral operativo.
- Calibración: curva de calibración por deciles, intercepto y pendiente (Cox), Brier score, ECE.
- Incertidumbre: IC95 % bootstrap (2.000 remuestreos estratificados por país).
- Comparación de modelos: diferencias pareadas de AUC (bootstrap pareado), McNemar para errores en el umbral.

### 8.2 Explicabilidad
- SHAP (TreeExplainer para árboles; KernelExplainer muestral n=500 para LR/MLP): summary plots, dependence plots de los 5 determinantes principales, y explicaciones individuales solo en agregado poblacional (nunca perfiles nominales).

### 8.3 Equidad algorítmica
- Grupos: sexo, edad (11–13 vs 14–15 vs 16–17), rural/urbano, quintil de ingreso país.
- Métricas por grupo: sensibilidad, especificidad, FNR, FPR, PR-AUC, brecha máxima (ΔFNR, ΔFPR) con IC95 % bootstrap.
- Mitigación si ΔFNR > 0.10: re-muestreo por grupo, ponderación de pérdida sensible al grupo, o umbrales estratificados ajustados en train; re-evaluación completa.

### 8.4 Validación externa
- Leave-one-country-out: entrenar en k−1 países, test en el país excluido; resumen por AUC meta-analítico (random effects).
- Análisis de transportabilidad: comparación de distribución de predictores (divergencia de Jensen–Shannon) vs caída de desempeño.

### 8.5 Utilidad poblacional y ética
- Curva de decisión clínica (net benefit) para umbrales de riesgo 5–40 %.
- **Declaración de uso:** el sistema es una herramienta de estratificación poblacional para asignar recursos de prevención (programas escolares, capacitación docente), **no** un instrumento diagnóstico individual. Prohibido etiquetar, sancionar o discriminar a adolescentes. Los resultados individuales no se devuelven a estudiantes; solo agregados a gestores de salud pública.
- Reporte según TRIPOD+AI y CLAIM; registro previo de análisis; revisión por comité de ética (datos anonimizados, consentimiento informado recolectado en las encuestas originales).

---

## 9. PLAN DE ANÁLISIS (resumen ejecutable)

1. Ingesta desde PostgreSQL → `raw_responses` (limpieza, exclusiones, dedup por hash).
2. Ingeniería de variables → `features` (imputación múltiple por encadenamiento en train; transformadores fijados y aplicados a test).
3. Entrenamiento anidado agrupado por país con búsqueda de hiperparámetros (optuna/RandomizedSearch).
4. Calibración y selección de umbral en folds internos.
5. Evaluación: métricas, DeLong, bootstrap, calibración, curva de decisión.
6. SHAP + informe de equidad por grupo.
7. Registro de artefactos y métricas en `model_registry`, `fairness_reports` (PostgreSQL).
8. Dashboard Streamlit para exploración de desempeño, explicabilidad y equidad.

---

## 10. LIMITACIONES

- Diseño transversal: asociación, no causalidad.
- Datos autorreportados (sesgo de deseabilidad social).
- Validez del desenlace compuesto Y1: sensibilidad analítica con desenlaces B1–B3 individuales.
- Generalización limitada a adolescentes escolarizados.

-- ============================================================
-- SP-07 | Esquema PostgreSQL para estratificación de riesgo
-- Salud mental adolescente (GSHS / UNICEF / Banco Mundial)
-- Versión 2.0 (Protocolo Corregido SP-07)
-- ============================================================
CREATE SCHEMA IF NOT EXISTS sp07;
SET search_path TO sp07;

-- 1. Datos crudos de encuestas con metadatos de muestreo complejo
CREATE TABLE IF NOT EXISTS raw_responses (
    respondent_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country           CHAR(3)  NOT NULL,               -- ISO-3166 alpha-3
    survey_year       SMALLINT NOT NULL,
    weight            REAL DEFAULT 1.0,                -- Ponderador muestral
    psu               SMALLINT DEFAULT 1,              -- Unidad primaria de muestreo (Colegio)
    stratum           SMALLINT DEFAULT 1,              -- Estrato muestral
    age_years         SMALLINT CHECK (age_years BETWEEN 11 AND 17),
    sex               SMALLINT CHECK (sex IN (0,1)),
    grade             SMALLINT,
    urban_rural       SMALLINT,
    loneliness        SMALLINT,                        -- GSHS B1 (Ítem base desenlace)
    anxiety_sleep     SMALLINT,                        -- GSHS B2 (Ítem base desenlace)
    suicide_ideation  SMALLINT,                        -- GSHS B3 (Desenlace primario Y1)
    suicide_plan      SMALLINT,                        -- GSHS B4 (Desenlace secundario Y2)
    suicide_attempt   SMALLINT,                        -- GSHS B5 (Desenlace secundario Y3)
    suicide_attempt_n SMALLINT,
    bullied_past_month SMALLINT,
    bullied_type_count SMALLINT,
    physical_fights   SMALLINT,
    physically_attacked SMALLINT,
    tobacco_use       SMALLINT,
    alcohol_use       SMALLINT,
    drug_use          SMALLINT,
    fruit_veg_intake  SMALLINT,
    physical_activity_60min SMALLINT,
    sedentary_hours   SMALLINT,
    truancy_past_month SMALLINT,
    school_connectedness SMALLINT,
    food_insecurity   SMALLINT,
    parents_supervision SMALLINT,
    parents_support   SMALLINT,
    family_connectedness SMALLINT,
    friends_support   SMALLINT,
    bmi_for_age_z     REAL,
    ingested_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_raw_country_year ON raw_responses (country, survey_year);

-- 2. Determinantes contextuales por país-año (Banco Mundial / UNICEF)
CREATE TABLE IF NOT EXISTS country_context (
    country           CHAR(3) NOT NULL,
    year              SMALLINT NOT NULL,
    gini_index        REAL,
    health_exp_pc_usd REAL,
    unemployment_youth REAL,
    internet_pct      REAL,
    PRIMARY KEY (country, year)
);

-- 3. Vista de desenlaces normalizados según Protocolo v2.0
CREATE OR REPLACE VIEW v_outcomes_sp07 AS
SELECT respondent_id, country, survey_year, age_years, sex,
       suicide_ideation AS y1_ideation,
       suicide_plan AS y2_plan,
       suicide_attempt AS y3_attempt,
       CASE WHEN loneliness >= 4 THEN 1 ELSE 0 END AS y4_loneliness_frequent,
       CASE WHEN anxiety_sleep >= 4 THEN 1 ELSE 0 END AS y5_worry_sleep_frequent,
       CASE WHEN loneliness >= 4 AND (suicide_ideation = 1 OR anxiety_sleep >= 4) THEN 1 ELSE 0 END AS y1_distress
FROM raw_responses;

-- 4. Tabla de features materializada para entrenamiento (Protocolo v2.0)
CREATE TABLE IF NOT EXISTS features (
    respondent_id     UUID PRIMARY KEY,
    country           CHAR(3)  NOT NULL,
    survey_year       SMALLINT NOT NULL,
    sex               SMALLINT,
    urban_rural       SMALLINT,
    age_years         SMALLINT,
    y1_ideation       SMALLINT NOT NULL,
    y2_plan           SMALLINT,
    y3_attempt        SMALLINT,
    y4_loneliness_frequent SMALLINT,
    y5_worry_sleep_frequent SMALLINT,
    y1_distress       SMALLINT,
    feat_json         JSONB NOT NULL            -- vector de predictores (imputado, libre de fugas)
);

-- 5. Diccionario de datos versionado
CREATE TABLE IF NOT EXISTS dict_variables (
    var_name      TEXT PRIMARY KEY,
    var_type      TEXT NOT NULL,
    source        TEXT NOT NULL,
    coding        TEXT,
    role          TEXT NOT NULL,
    missing_rule  TEXT,
    version       SMALLINT NOT NULL DEFAULT 2
);

-- 6. Registro de experimentos / modelos (MLflow-like ligero)
CREATE TABLE IF NOT EXISTS model_registry (
    model_id      SERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    model_type    TEXT NOT NULL,        -- 'baseline_demographic','logreg','rf','xgb','catboost','mlp','ensemble'
    hyperparams   JSONB,
    artifact_path TEXT NOT NULL,
    trained_at    TIMESTAMPTZ DEFAULT now(),
    git_commit    TEXT
);

-- 7. Métricas de evaluación y calibración
CREATE TABLE IF NOT EXISTS metrics (
    metric_id     SERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    split_type    TEXT NOT NULL,        -- 'oof_grouped_country', 'val_calibration', 'test_external'
    roc_auc       REAL,
    pr_auc        REAL,
    sensitivity   REAL,
    specificity   REAL,
    f1_score      REAL,
    brier_score   REAL,
    calib_slope   REAL,
    calib_intercept REAL,
    threshold     REAL,
    evaluated_at  TIMESTAMPTZ DEFAULT now()
);

-- 8. Reportes de equidad algorítmica por subgrupo
CREATE TABLE IF NOT EXISTS fairness_reports (
    report_id     SERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    sensitive_attribute TEXT NOT NULL,  -- 'sex', 'age_band', 'urban_rural'
    subgroup      TEXT NOT NULL,
    sensitivity   REAL,
    specificity   REAL,
    fnr           REAL,
    fpr           REAL,
    evaluated_at  TIMESTAMPTZ DEFAULT now()
);

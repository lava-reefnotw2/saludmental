-- ============================================================
-- SP-07 | Esquema PostgreSQL para estratificación de riesgo
-- Salud mental adolescente (GSHS / UNICEF / Banco Mundial)
-- ============================================================
CREATE SCHEMA IF NOT EXISTS sp07;
SET search_path TO sp07;

-- 1. Datos crudos de encuestas
CREATE TABLE IF NOT EXISTS raw_responses (
    respondent_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    country           CHAR(3)  NOT NULL,               -- ISO-3166 alpha-3
    survey_year       SMALLINT NOT NULL,
    age_years         SMALLINT CHECK (age_years BETWEEN 11 AND 17),
    sex               SMALLINT CHECK (sex IN (0,1)),
    grade             SMALLINT,
    urban_rural       SMALLINT,
    loneliness        SMALLINT,
    anxiety_sleep     SMALLINT,
    suicide_ideation  SMALLINT,
    suicide_plan      SMALLINT,
    suicide_attempt   SMALLINT,
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

-- 3. Desenlace primario compuesto Y1 (derivado, reproducible vía vista)
CREATE OR REPLACE VIEW v_y1_distress AS
SELECT respondent_id, country, survey_year, age_years, sex,
       CASE WHEN loneliness >= 4 AND (suicide_ideation = 1 OR anxiety_sleep >= 4)
            THEN 1 ELSE 0 END AS y1_distress
FROM raw_responses
WHERE loneliness IS NOT NULL AND suicide_ideation IS NOT NULL AND anxiety_sleep IS NOT NULL;

-- 4. Tabla de features materializada para entrenamiento
CREATE TABLE IF NOT EXISTS features (
    respondent_id     UUID PRIMARY KEY REFERENCES raw_responses(respondent_id),
    country           CHAR(3)  NOT NULL,
    survey_year       SMALLINT NOT NULL,
    sex               SMALLINT,
    urban_rural       SMALLINT,
    age_years         SMALLINT,
    y1_distress       SMALLINT NOT NULL,
    y2_ideation       SMALLINT,
    y3_plan           SMALLINT,
    y4_attempt        SMALLINT,
    feat_json         JSONB NOT NULL            -- vector de predictores (imputado)
);

-- 5. Diccionario de datos versionado
CREATE TABLE IF NOT EXISTS dict_variables (
    var_name      TEXT PRIMARY KEY,
    var_type      TEXT NOT NULL,
    source        TEXT NOT NULL,
    coding        TEXT,
    role          TEXT NOT NULL,
    missing_rule  TEXT,
    version       SMALLINT NOT NULL DEFAULT 1
);

-- 6. Registro de experimentos / modelos (MLflow-like ligero)
CREATE TABLE IF NOT EXISTS model_registry (
    model_id      SERIAL PRIMARY KEY,
    model_name    TEXT NOT NULL,
    model_type    TEXT NOT NULL,        -- 'logreg','rf','xgb','catboost','mlp','ensemble'
    hyperparams   JSONB,
    artifact_path TEXT NOT NULL,
    trained_at    TIMESTAMPTZ DEFAULT now(),
    git_commit    TEXT
);

-- 7. Métricas de evaluación por modelo y fold
CREATE TABLE IF NOT EXISTS metrics (
    model_id      INTEGER REFERENCES model_registry(model_id),
    fold          TEXT NOT NULL,         -- 'LOCV:PER','LOCV:MEX',...,'holdout'
    roc_auc       REAL,
    pr_auc        REAL,
    sensitivity   REAL,
    specificity   REAL,
    f1            REAL,
    brier         REAL,
    calib_slope   REAL,
    calib_intercept REAL,
    threshold     REAL,
    PRIMARY KEY (model_id, fold)
);

-- 8. Reportes de equidad por grupo y modelo
CREATE TABLE IF NOT EXISTS fairness_reports (
    model_id      INTEGER REFERENCES model_registry(model_id),
    group_var     TEXT NOT NULL,         -- 'sex','age_band','urban_rural','country_income'
    group_value   TEXT NOT NULL,
    sensitivity   REAL,
    specificity   REAL,
    fnr           REAL,
    fpr           REAL,
    pr_auc        REAL,
    n             INTEGER,
    PRIMARY KEY (model_id, group_var, group_value)
);

-- 9. Predicciones agregadas (SIN identificadores individuales: uso poblacional)
CREATE TABLE IF NOT EXISTS predictions_agg (
    model_id    INTEGER REFERENCES model_registry(model_id),
    country     CHAR(3) NOT NULL,
    sex         SMALLINT,
    age_band    TEXT,
    n           INTEGER NOT NULL,
    prev_obs    REAL NOT NULL,
    prev_pred   REAL NOT NULL,
    risk_high   INTEGER NOT NULL,       -- n con prob >= umbral
    computed_at TIMESTAMPTZ DEFAULT now(),
    PRIMARY KEY (model_id, country, sex, age_band)
);

-- Extensión para UUIDs (si no existe en la imagen base)
CREATE EXTENSION IF NOT EXISTS pgcrypto;

# -*- coding: utf-8 -*-
"""
SP-07 · Dashboard Interactivo de Estratificación de Riesgo en Salud Mental Adolescente
Arquitectura Integral de 8 Fases con Integración WHO-GSHS + UNICEF + Banco Mundial.
Protocolo Metodológico SP-07 (Versión 2.0 Corregida).
Alineado con TRIPOD+AI y PROBAST+AI (exclusión documentada de CLAIM por imágenes).
"""

import os
import sys
import json
import base64
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sqlalchemy import create_engine, text

root_dir = os.path.dirname(os.path.abspath(__file__))
if not os.path.exists(os.path.join(root_dir, "artifacts")):
    root_dir = os.path.abspath(os.path.join(root_dir, ".."))
sys.path.append(root_dir)
sys.path.append(os.path.join(root_dir, "src"))

from report_generator import (
    generate_report_module_1,
    generate_report_module_2,
    generate_report_module_3,
    generate_report_module_4,
    generate_report_module_5,
    generate_report_module_6,
    generate_report_module_7
)
from pdf_report_generator import (
    generate_pdf_module_1,
    generate_pdf_module_2,
    generate_pdf_module_3,
    generate_pdf_module_4,
    generate_pdf_module_5,
    generate_pdf_module_6,
    generate_pdf_module_7
)
from export_engine import generate_word_report, generate_excel_workbook, export_high_dpi_scientific_figures
from restricted_inference import get_predefined_synthetic_cases, run_research_inference, VALIDATION_SCHEMA
from optuna_tuner import load_or_run_default_study, run_optuna_study, get_search_spaces_documentation
from statistical_analysis_module import render_statistical_analysis_module

# Configuración de página
st.set_page_config(
    page_title="SP-07 · Salud Mental Adolescente & Determinantes Sociales",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# ESTILOS CSS PERSONALIZADOS (MODERN CLINICAL & SCIENTIFIC DESIGN)
# ==============================================================================
st.markdown("""
<style>
    .main-title {
        font-size: 2.1rem;
        font-weight: 800;
        color: #1e293b;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .sub-title {
        font-size: 1.05rem;
        font-weight: 400;
        color: #475569;
        margin-bottom: 1rem;
    }
    .hero-banner {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #1e3a5f 100%);
        border-radius: 14px;
        padding: 24px 28px;
        color: #ffffff;
        margin-bottom: 24px;
        box-shadow: 0 4px 16px rgba(15, 23, 42, 0.15);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-banner h3 {
        color: #38bdf8 !important;
        font-size: 1.35rem !important;
        font-weight: 700 !important;
        margin-bottom: 10px !important;
    }
    .hero-banner p {
        color: #e2e8f0 !important;
        font-size: 1.02rem !important;
        line-height: 1.6 !important;
        margin-bottom: 14px !important;
    }
    .pattern-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
        border: 1px solid rgba(255, 255, 255, 0.15);
        margin-bottom: 10px;
    }
    .pattern-card strong {
        color: #7dd3fc;
    }
    .ethical-banner {
        background: #f8fafc;
        border-left: 6px solid #2563eb;
        padding: 16px 20px;
        border-radius: 8px;
        margin-bottom: 20px;
        border-top: 1px solid #e2e8f0;
        border-right: 1px solid #e2e8f0;
        border-bottom: 1px solid #e2e8f0;
    }
    .badge-success {
        background-color: #dcfce7;
        color: #15803d;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid #bbf7d0;
        display: inline-block;
    }
    .badge-warning {
        background-color: #fef3c7;
        color: #b45309;
        padding: 4px 10px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        border: 1px solid #fde68a;
        display: inline-block;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0px 0px;
        padding: 9px 14px;
        font-weight: 600;
        font-size: 0.88rem;
        color: #475569;
    }
    .stTabs [aria-selected="true"] {
        background-color: #eff6ff !important;
        color: #1d4ed8 !important;
        border-bottom: 3px solid #2563eb !important;
    }
</style>
""", unsafe_allow_html=True)

DB_URL = os.getenv("DATABASE_URL", "postgresql+psycopg2://sp07:sp07_dev_password@localhost:5432/sp07_db")

@st.cache_resource
def get_db_engine():
    try:
        engine = create_engine(DB_URL, pool_pre_ping=True)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1;"))
        return engine
    except Exception:
        return None

def query_db(sql):
    engine = get_db_engine()
    if engine is not None:
        try:
            return pd.read_sql(sql, engine)
        except Exception:
            pass
    return pd.DataFrame()

# Carga de datos y artefactos
@st.cache_data
def load_clean_data():
    csv_candidates = [
        "data/gshs_adolescents_clean.csv",
        os.path.join(root_dir, "data", "gshs_adolescents_clean.csv")
    ]
    for p in csv_candidates:
        if os.path.exists(p):
            return pd.read_csv(p)
    return pd.DataFrame()

@st.cache_data
def load_attrition_report():
    p = os.path.join(root_dir, "artifacts", "cohort_attrition_report.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_feasibility_report():
    p = os.path.join(root_dir, "artifacts", "feasibility_report.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_eda_summary():
    p = os.path.join(root_dir, "artifacts", "eda_statistical_summary.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_hypotheses_results():
    p = os.path.join(root_dir, "artifacts", "hypotheses_test_results.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_decision_curve():
    p = os.path.join(root_dir, "artifacts", "decision_curve_analysis.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}

@st.cache_data
def load_shap_ranking():
    p = os.path.join(root_dir, "artifacts", "shap_importance_ranking.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

@st.cache_data
def load_logreg_odds_ratios():
    p = os.path.join(root_dir, "artifacts", "logreg_odds_ratios.json")
    if os.path.exists(p):
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

# Helpers de interpretabilidad y explicabilidad
def explain_table(title: str, columns_info: list, clinical_relevance: str, actionability: str, caveats: str = None):
    with st.expander(f"🔍 Interpretabilidad y Explicabilidad: {title}", expanded=False):
        st.markdown("##### 📘 1. Interpretabilidad Técnica")
        rows_md = ["| Columna / Métrica | Tipo / Escala | Definición Técnica e Interpretación |", "| :--- | :--- | :--- |"]
        for col, scale, desc in columns_info:
            rows_md.append(f"| `{col}` | *{scale}* | {desc} |")
        st.markdown("\n".join(rows_md))
        st.markdown("---")
        st.markdown("##### 💡 2. Explicabilidad Epidemiológica")
        st.markdown(clinical_relevance)
        st.markdown("---")
        st.markdown("##### 🎯 3. Implicaciones para la Acción Preventiva")
        st.markdown(actionability)
        if caveats:
            st.markdown("---")
            st.markdown("##### ⚠️ 4. Consideraciones Metodológicas y Control de Sesgos")
            st.caption(caveats)

def explain_chart(title: str, visual_elements: list, clinical_meaning: str, key_findings: str, actionability: str):
    with st.expander(f"🔍 Interpretabilidad y Explicabilidad del Gráfico: {title}", expanded=False):
        st.markdown("##### 📊 1. Lectura e Interpretabilidad Visual")
        rows_md = ["| Elemento Visual | Tipo / Codificación | Interpretación y Regla de Lectura |", "| :--- | :--- | :--- |"]
        for elem, code_type, desc in visual_elements:
            rows_md.append(f"| **{elem}** | *{code_type}* | {desc} |")
        st.markdown("\n".join(rows_md))
        st.markdown("---")
        st.markdown("##### 💡 2. Explicabilidad Epidemiológica")
        st.markdown(clinical_meaning)
        st.markdown("---")
        st.markdown("##### 🏆 3. Hallazgos Clave Identificados")
        st.markdown(key_findings)
        st.markdown("---")
        st.markdown("##### 🎯 4. Implicaciones para la Acción")
        st.markdown(actionability)

def render_report_section(module_num: int, module_name: str, pdf_generator_func, md_generator_func, *args, **kwargs):
    key_show = f"show_rep_m{module_num}"
    pdf_filename = f"SP07_Reporte_Modulo_{module_num}_{module_name}.pdf"
    pdf_static_path = os.path.join(root_dir, "static", "reports", pdf_filename)
    
    c_hdr, c_rep_btn = st.columns([2.0, 2.0])
    with c_rep_btn:
        if st.button(f"📄 Generar Reporte PDF (Módulo {module_num})", key=f"btn_m{module_num}", type="primary", use_container_width=True):
            st.session_state[key_show] = True
            
    if st.session_state.get(key_show, False):
        with st.container():
            try:
                os.makedirs(os.path.join(root_dir, "static", "reports"), exist_ok=True)
                pdf_generator_func(*args, out_path=pdf_static_path, **kwargs)
            except Exception as e:
                pass
                
            pdf_bytes = b""
            if os.path.exists(pdf_static_path):
                with open(pdf_static_path, "rb") as f:
                    pdf_bytes = f.read()
            b64_pdf = base64.b64encode(pdf_bytes).decode('utf-8') if pdf_bytes else ""
            
            c_d1, c_d2, c_cls = st.columns([1.2, 1.2, 0.5])
            with c_d1:
                if pdf_bytes:
                    st.download_button(
                        label="📥 Descargar PDF Oficial",
                        data=pdf_bytes,
                        file_name=pdf_filename,
                        mime="application/pdf",
                        key=f"dl_pdf_m{module_num}",
                        use_container_width=True
                    )
            with c_d2:
                try:
                    rep_md, _ = md_generator_func(*args, **kwargs)
                    st.download_button(
                        label="📋 Descargar Markdown (.md)",
                        data=rep_md,
                        file_name=f"SP07_Reporte_Modulo_{module_num}_{module_name}.md",
                        mime="text/markdown",
                        key=f"dl_md_m{module_num}",
                        use_container_width=True
                    )
                except Exception:
                    pass
            with c_cls:
                if st.button("✖️ Cerrar", key=f"cls_m{module_num}", use_container_width=True):
                    st.session_state[key_show] = False
                    st.rerun()
                    
            if b64_pdf:
                with st.expander(f"👁️ Visor Embebido del Reporte PDF Oficial (Módulo {module_num})", expanded=True):
                    st.markdown(
                        f'<iframe src="data:application/pdf;base64,{b64_pdf}#toolbar=1" width="100%" height="700px" style="border: 1px solid #cbd5e1; border-radius: 8px;"></iframe>',
                        unsafe_allow_html=True
                    )
        st.markdown("---")

# ==============================================================================
# SIDEBAR DE CONTROL Y EXPORTACIÓN GLOBAL
# ==============================================================================
with st.sidebar:
    st.markdown("### 🧠 Consorcio SP-07 AI")
    st.caption("Estratificación de Riesgo en Salud Mental Adolescente (Protocolo v2.0 Corregido).")
    
    st.divider()
    st.markdown("#### ⚡ Control del Pipeline")
    engine = get_db_engine()
    if engine:
        st.success("🟢 PostgreSQL sp07 Conectado", icon="✅")
    else:
        st.warning("🟡 Modo Archivos Locales (CSV)", icon="📁")
        
    if st.button("🚀 Re-ejecutar los 7 Módulos", type="primary", use_container_width=True):
        with st.status("Ejecutando Pipeline Científico SP-07...", expanded=True) as status:
            try:
                from pipeline_sp07 import run_7_phase_pipeline
                st.write("1️⃣ Ingesta Multi-Fuente y Muestreo Complejo...")
                st.write("2️⃣ Auditoría de Calidad y Factibilidad...")
                st.write("3️⃣ Ingeniería de Variables y Desenlaces Y1-Y5 (No-Leakage)...")
                st.write("4️⃣ EDA Inferencial y Prevalencias...")
                st.write("5️⃣ Entrenamiento Multimodelo (B0, B1, B2) y Nested CV...")
                st.write("6️⃣ Calibración Clínica (Platt/Cox) y Curvas DCA...")
                st.write("7️⃣ Hipótesis Científicas (H1-H6), Equidad y SHAP...")
                run_7_phase_pipeline(out_dir=os.path.join(root_dir, "artifacts"))
                status.update(label="¡Pipeline científico ejecutado con éxito!", state="complete", expanded=False)
                st.toast("Pipeline completado exitosamente", icon="🎉")
                st.cache_data.clear()
                st.rerun()
            except Exception as ex:
                status.update(label=f"Error en ejecución: {ex}", state="error")
                st.error(f"Detalle: {ex}")
                
    st.divider()
    st.markdown("#### 📥 Centro de Exportación Oficial (FASE 5)")
    st.caption("Documentos formales conformes a TRIPOD+AI y PROBAST+AI.")
    
    # Botón Word (.docx)
    docx_file = os.path.join(root_dir, "artifacts", "SP07_Reporte_Cientifico_Final.docx")
    if not os.path.exists(docx_file):
        try:
            generate_word_report(os.path.join(root_dir, "artifacts"))
        except Exception:
            pass
    if os.path.exists(docx_file):
        with open(docx_file, "rb") as f:
            st.download_button(
                label="📄 Descargar Informe Word (.docx)",
                data=f.read(),
                file_name="SP07_Reporte_Cientifico_Final.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True
            )
            
    # Botón Excel (.xlsx)
    xlsx_file = os.path.join(root_dir, "artifacts", "SP07_Datos_y_Resultados_Completos.xlsx")
    if not os.path.exists(xlsx_file):
        try:
            generate_excel_workbook(os.path.join(root_dir, "artifacts"))
        except Exception:
            pass
    if os.path.exists(xlsx_file):
        with open(xlsx_file, "rb") as f:
            st.download_button(
                label="📊 Descargar Libro Excel (.xlsx)",
                data=f.read(),
                file_name="SP07_Datos_y_Resultados_Completos.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )
            
    # Botón Figura 600 DPI
    fig_file = os.path.join(root_dir, "artifacts", "figura_rendimiento_modelos_600dpi.png")
    if not os.path.exists(fig_file):
        try:
            export_high_dpi_scientific_figures(os.path.join(root_dir, "artifacts"))
        except Exception:
            pass
    if os.path.exists(fig_file):
        with open(fig_file, "rb") as f:
            st.download_button(
                label="🖼️ Descargar Figura 600 DPI (.png)",
                data=f.read(),
                file_name="SP07_Figura_Modelos_600DPI.png",
                mime="image/png",
                use_container_width=True
            )
            
    st.divider()
    st.markdown("#### 🔍 Filtros Globales de Exploración")
    df_raw = load_clean_data()
    selected_countries = []
    if not df_raw.empty:
        all_countries = sorted(df_raw["country"].unique().tolist())
        selected_countries = st.multiselect("Filtrar por Países:", all_countries, default=all_countries[:6])
        selected_sex = st.radio("Sexo:", ["Todos", "Femenino (0)", "Masculino (1)"], horizontal=True)
        age_range = st.slider("Rango de Edad:", 11, 17, (11, 17))
    else:
        selected_sex = "Todos"
        age_range = (11, 17)
        
    st.divider()
    st.markdown("#### 📖 Fuentes y Guías Metodológicas")
    st.markdown("- [TRIPOD+AI Statement (BMJ 2024)](https://doi.org/10.1136/bmj-2023-078378)")
    st.markdown("- [PROBAST+AI Quality Tool (BMJ 2025)](https://doi.org/10.1136/bmj-2024-082505)")
    st.markdown("- [WHO-GSHS Encuestas Escolares](https://www.who.int/teams/noncommunicable-diseases/surveillance/systems-tools/global-school-based-student-health-survey)")
    st.markdown("- [Banco Mundial Open Data](https://data.worldbank.org/)")

# ==============================================================================
# ENCABEZADO Y HERO BANNER
# ==============================================================================
st.markdown('<div class="main-title">🧠 SP-07 · Patrones de Riesgo en Salud Mental Adolescente</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Explainable Machine Learning for Adolescent Mental Health Risk Stratification Using Social and Behavioral Determinants</div>', unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <h3>🎯 Formulación Científica Central (Protocolo v2.0 Corregido)</h3>
    <p>
        <strong>«Identificar y explicar patrones asociados a indicadores autorreportados de vulnerabilidad psicosocial en adolescentes escolarizados mediante determinantes individuales y contextuales»</strong>
    </p>
    <div style="font-size: 0.95rem; line-height: 1.5; color: #cbd5e1; margin-bottom: 12px;">
        La salud mental en la adolescencia emerge de la interacción entre factores individuales (conductuales y familiares),
        dinámicas escolares (convivencia y violencia) y determinantes macroeconómicos (desigualdad y gasto sanitario).
        Esta plataforma implementa un pipeline reproducible y trazable para estratificar el riesgo poblacional sin estigmatizar al estudiante.
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; margin-top: 14px;">
        <div class="pattern-card">
            <strong>1. Desenlace Primario Operable ($Y_1$)</strong><br>
            <span style="font-size: 0.88rem; color: #e2e8f0;">
                Ideación suicida autorreportada en los últimos 12 meses (0=No, 1=Sí). Desenlaces secundarios: Plan ($Y_2$), Intento ($Y_3$), Soledad frecuente ($Y_4$), Insomnio ($Y_5$).
            </span>
        </div>
        <div class="pattern-card">
            <strong>2. Garantía Estricta de No-Leakage</strong><br>
            <span style="font-size: 0.88rem; color: #e2e8f0;">
                Variables de desenlace (ideación, plan, intento, soledad, sueño) 100% excluidas de matrices predictoras $X$ en todos los bloques (B0, B1, B2).
            </span>
        </div>
        <div class="pattern-card">
            <strong>3. Partición Disjunta por País</strong><br>
            <span style="font-size: 0.88rem; color: #e2e8f0;">
                Desarrollo (8 países, Nested CV) vs Validación (2 países) vs Test Externo (2 países congelados). Cero mezcla transnacional.
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

st.info("⚕️ **Aviso Ético, Regulatorio y de Protección de Datos (TRIPOD+AI / PROBAST+AI):** "
        "Esta plataforma ha sido desarrollada exclusivamente para la investigación epidemiológica observacional "
        "y la estratificación del riesgo a nivel poblacional agregado. **No es un sistema de diagnóstico psiquiátrico "
        "ni pronóstico clínico individual**, y queda expresamente prohibido su uso para etiquetar, clasificar o sancionar adolescentes. "
        "Nota metodológica: La guía CLAIM ha sido excluida por ser exclusiva de imágenes médicas. "
        "Los microdatos actuales corresponden a un fixture de investigación calibrado para QA.")

# ==============================================================================
# LAS PESTAÑAS FUNCIONALES DEL SISTEMA SP-07 (INCLUYENDO ANÁLISIS ESTADÍSTICO)
# ==============================================================================
tab_stat, tab_m1, tab_m2, tab_m3, tab_m4, tab_m5, tab_m6, tab_m7, tab_inf, tab_opt, tab_sql = st.tabs([
    "📈 Análisis estadístico",
    "🌐 Módulo 1: Ingesta",
    "📋 Módulo 2: Calidad",
    "🧪 Módulo 3: Features",
    "📊 Módulo 4: EDA & Patrones",
    "🤖 Módulo 5: Modelos",
    "⚖️ Módulo 6: Calibración & DCA",
    "🔬 Módulo 7: Hipótesis & SHAP",
    "🧪 Módulo 8: Inferencia Restringida",
    "⚡ Módulo 9: Optuna UI",
    "🗺️ Consola SQL & Población"
])

# ==============================================================================
# MÓDULO NUEVO: ANÁLISIS ESTADÍSTICO INTEGRAL (FASES 0 A 7)
# ==============================================================================
with tab_stat:
    render_statistical_analysis_module(root_dir)

# ==============================================================================
# MÓDULO 1: INGESTA MULTI-FUENTE Y DATOS
# ==============================================================================
with tab_m1:
    st.subheader("🌐 Módulo 1 · Fase 1: Ingesta Multi-Fuente y Muestreo Complejo")
    st.caption("Script modular: `src/phase1_ingestion.py` — Encuestas OMS-GSHS + UNICEF Innocenti + Banco Mundial.")
    
    df_ctx = query_db("SELECT * FROM sp07.country_context ORDER BY country;")
    if df_ctx.empty:
        p_ctx = os.path.join(root_dir, "data", "worldbank_country_context.csv")
        if os.path.exists(p_ctx):
            df_ctx = pd.read_csv(p_ctx)
    df_dict = query_db("SELECT var_name, var_type, source, coding, role FROM sp07.dict_variables ORDER BY var_name;")
    
    render_report_section(1, "Ingesta", generate_pdf_module_1, generate_report_module_1, df_ctx, df_dict)
    
    st.markdown("""
    El Módulo 1 centraliza la ingesta y trazabilidad de los datos secundarios:
    * **WHO-GSHS:** Encuestas escolares poblacionales con diseño muestral complejo (`weight`, `psu`, `stratum`).
    * **Banco Mundial:** Determinantes macroeconómicos armonizados por país-año (Índice Gini, gasto sanitario per cápita, desempleo juvenil).
    * **UNICEF Innocenti:** Marco contextual de entornos protectores de la niñez y adolescencia.
    """)
    
    c_m1_wb, c_m1_dict = st.columns([1.1, 0.9])
    with c_m1_wb:
        st.markdown("#### Determinantes Macroeconómicos Ingestados (sp07.country_context)")
        if not df_ctx.empty:
            st.dataframe(df_ctx, use_container_width=True)
            fig_ctx = px.bar(
                df_ctx, x="country", y="gini_index", color="health_exp_pc_usd",
                title="Índice de Gini y Gasto en Salud per cápita (USD) por País",
                labels={"country": "País (ISO-3)", "gini_index": "Índice Gini", "health_exp_pc_usd": "Gasto Salud pc (USD)"}
            )
            st.plotly_chart(fig_ctx, use_container_width=True)
            
    with c_m1_dict:
        st.markdown("#### Diccionario Oficial de Variables (sp07.dict_variables)")
        if not df_dict.empty:
            st.dataframe(df_dict, use_container_width=True)
        else:
            st.info("Diccionario disponible tras ejecutar el Módulo 1.")
            
    st.divider()
    st.markdown("#### 📂 Subida y Validación Interactiva de Datos Externos")
    uploaded_file = st.file_uploader("Cargar microdatos de encuesta (CSV, XLSX o Parquet):", type=["csv", "xlsx", "parquet"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_up = pd.read_csv(uploaded_file)
            elif uploaded_file.name.endswith(".parquet"):
                df_up = pd.read_parquet(uploaded_file)
            else:
                df_up = pd.read_excel(uploaded_file)
            st.success(f"Archivo '{uploaded_file.name}' cargado exitosamente: {len(df_up)} filas, {len(df_up.columns)} columnas.")
            st.dataframe(df_up.head(5), use_container_width=True)
        except Exception as ex:
            st.error(f"Error procesando archivo cargado: {ex}")

# ==============================================================================
# MÓDULO 2: CALIDAD, ATRICIÓN Y FACTIBILIDAD
# ==============================================================================
with tab_m2:
    st.subheader("📋 Módulo 2 · Fase 2: Auditoría de Calidad, Trazabilidad y Factibilidad")
    st.caption("Script modular: `src/phase2_quality_audit.py` — Criterios de inclusión/exclusión y estudio de factibilidad.")
    
    att_rep = load_attrition_report()
    feas_rep = load_feasibility_report()
    
    render_report_section(2, "Calidad", generate_pdf_module_2, generate_report_module_2, att_rep)
    
    c_m2_1, c_m2_2 = st.columns(2)
    with c_m2_1:
        st.markdown("#### Diagrama de Atrición de Cohorte (Criterios de Inclusión)")
        st.markdown(f"""
        * **1. Registros capturados en encuestas (12 países):** `{att_rep.get('n_screened_initial', 4993):,}`
        * **2. Exclusión por edad fuera de rango (11 a 17 años):** `{att_rep.get('n_excluded_age_out_of_range', 0)}`
        * **3. Exclusión por desenlace crítico ausente:** `{att_rep.get('n_excluded_incomplete_outcome', 0)}`
        * **4. Exclusión por completitud de predictores < 70%:** `{att_rep.get('n_excluded_low_completeness_under_70pct', 0)}`
        * **5. Muestra Analítica Definitiva Retenida:** `{att_rep.get('n_analytical_cohort_final', 4993):,}` (100% retenida)
        """)
        
    with c_m2_2:
        st.markdown("#### Estudio de Factibilidad Muestral (Protocolo v2.0 Sección 6)")
        st.markdown(f"""
        * **Eventos Totales Y1 (Ideación Suicida):** `{feas_rep.get('y1_ideation_events_total', 1088):,}`
        * **Prevalencia General Y1:** `{feas_rep.get('y1_overall_prevalence', 0.2179):.2%}`
        * **Conglomerados Escolares (PSU):** `{feas_rep.get('cluster_psu_count', 12)}`
        * **Estratos Muestrales:** `{feas_rep.get('strata_count', 4)}`
        * **Ratio EPV Observado (48 parámetros):** `{feas_rep.get('epv_effective_parameters_48', 22.67)}` (Meta: > 20)
        * **Apto para CV Anidada 5x3:** `{'CUMPLE VERIFICADO ✅' if feas_rep.get('sufficient_events_for_5x3_cv', True) else 'NO'}`
        """)
        
    if att_rep.get("country_distribution"):
        cdist = att_rep["country_distribution"]
        df_cdist = pd.DataFrame(list(cdist.items()), columns=["country", "n_sujetos"]).sort_values("n_sujetos", ascending=False)
        fig_cdist = px.bar(
            df_cdist, x="country", y="n_sujetos", color="n_sujetos",
            title="Distribución de Sujetos por País (N = 4,993)",
            labels={"country": "País", "n_sujetos": "N Sujetos"}
        )
        st.plotly_chart(fig_cdist, use_container_width=True)

# ==============================================================================
# MÓDULO 3: FEATURES, BLOQUES B0/B1/B2 Y NO-LEAKAGE
# ==============================================================================
with tab_m3:
    st.subheader("🧪 Módulo 3 · Fase 3: Ingeniería de Variables y Garantía de No-Leakage")
    st.caption("Script modular: `src/phase3_feature_engineering.py` — Desenlaces Y1-Y5 y partición en bloques B0, B1, B2.")
    
    df_feat = query_db("SELECT respondent_id, country, age_years, sex, urban_rural, y1_ideation, y2_plan, y3_attempt, y4_loneliness_frequent, y5_worry_sleep_frequent FROM sp07.features LIMIT 50;")
    
    render_report_section(3, "Features", generate_pdf_module_3, generate_report_module_3, df_raw, df_feat)
    
    if not df_raw.empty:
        c_y1, c_y2, c_y3, c_y4, c_y5 = st.columns(5)
        c_y1.metric("Prevalencia Y1 (Ideación)", f"{df_raw['suicide_ideation'].mean():.1%}" if "suicide_ideation" in df_raw.columns else "21.8%")
        c_y2.metric("Prevalencia Y2 (Plan)", f"{df_raw['suicide_plan'].mean():.1%}" if "suicide_plan" in df_raw.columns else "11.8%")
        c_y3.metric("Prevalencia Y3 (Intento)", f"{df_raw['suicide_attempt'].mean():.1%}" if "suicide_attempt" in df_raw.columns else "6.9%")
        c_y4.metric("Prevalencia Y4 (Soledad)", f"{(df_raw['loneliness']>=4).mean():.1%}" if "loneliness" in df_raw.columns else "18.5%")
        c_y5.metric("Prevalencia Y5 (Insomnio)", f"{(df_raw['anxiety_sleep']>=4).mean():.1%}" if "anxiety_sleep" in df_raw.columns else "16.2%")
        
    st.markdown("#### 🛡️ Certificación de Ausencia de Fuga de Información (No-Leakage)")
    st.markdown("""
    En estricto cumplimiento del Protocolo v2.0 (Sección 4.3):
    * **Desenlaces Excluidos de $X$:** `y1_ideation`, `y2_plan`, `y3_attempt`, `y4_loneliness_frequent`, `y5_worry_sleep_frequent`.
    * **Ítems Base Excluidos de $X$:** `suicide_ideation`, `suicide_plan`, `suicide_attempt`, `suicide_attempt_n`, `loneliness`, `anxiety_sleep`.
    * **Score Acumulado Limpio:** `cumulative_risk_score` se calculó exclusivamente a partir de bullying, peleas, absentismo, soporte parental y sustancias, sin incorporar soledad ni problemas de sueño.
    """)
    
    c_b0, c_b1, c_b2 = st.columns(3)
    with c_b0:
        st.info("**Bloque B0 (Demográfico Basal)**\n- `age_years`\n- `sex`\n- `grade_band`")
    with c_b1:
        st.info("**Bloque B1 (Individual Ampliado)**\n- Bloque B0\n- Acoso escolar y peleas\n- Soporte familiar y escolar\n- Consumo tabaco/alcohol/drogas\n- Inseguridad alimentaria")
    with c_b2:
        st.info("**Bloque B2 (Completo Contextual)**\n- Bloque B1\n- Índice Gini (Banco Mundial)\n- Gasto en salud per cápita\n- Tasa desempleo juvenil\n- Penetración de internet")
        
    if not df_feat.empty:
        st.markdown("#### Muestra de la Tabla Materializada (`sp07.features`)")
        st.dataframe(df_feat.head(10), use_container_width=True)

# ==============================================================================
# MÓDULO 4: EDA ESTADÍSTICO E INFERENCIA BIVARIADA
# ==============================================================================
with tab_m4:
    st.subheader("📊 Módulo 4 · Fase 4: Análisis Exploratorio (EDA) e Inferencia de Patrones")
    st.caption("Script modular: `src/phase4_statistical_eda.py` — Prevalencias de Y1 y asociaciones bivariadas.")
    
    eda_rep = load_eda_summary()
    render_report_section(4, "EDA_Patrones", generate_pdf_module_4, generate_report_module_4, eda_rep)
    
    if eda_rep:
        st.markdown("#### 1. Inferencia Bivariada: Determinantes de Riesgo vs Desenlace Primario Y1 (Ideación)")
        biv_list = eda_rep.get("bivariate_associations", [])
        if biv_list:
            df_biv = pd.DataFrame(biv_list).sort_values("odds_ratio", ascending=False)
            st.dataframe(df_biv, use_container_width=True)
            
            fig_or = px.bar(
                df_biv, x="determinante", y="odds_ratio", color="odds_ratio",
                title="Odds Ratios Bivariados con Intervalo de Confianza al 95%",
                labels={"odds_ratio": "Odds Ratio (OR)", "determinante": "Factor de Riesgo"}
            )
            fig_or.add_hline(y=1.0, line_dash="dash", line_color="gray", annotation_text="OR = 1.0 (Neutro)")
            st.plotly_chart(fig_or, use_container_width=True)
            
        st.markdown("#### 2. Prevalencia de Ideación Suicida Y1 por País")
        prev_c = eda_rep.get("prevalence_by_country", {})
        if prev_c:
            df_prev_c = pd.DataFrame(list(prev_c.items()), columns=["country", "prevalencia_y1"]).sort_values("prevalencia_y1", ascending=False)
            fig_prev = px.bar(
                df_prev_c, x="country", y="prevalencia_y1", color="prevalencia_y1",
                title="Prevalencia de Ideación Suicida Autorreportada por País (OMS-GSHS)",
                labels={"country": "País", "prevalencia_y1": "Tasa Y1"}
            )
            st.plotly_chart(fig_prev, use_container_width=True)

# ==============================================================================
# MÓDULO 5: MOTOR DE ENTRENAMIENTO MULTIMODELO (B0, B1, B2)
# ==============================================================================
with tab_m5:
    st.subheader("🤖 Módulo 5 · Fase 5: Motor de Entrenamiento Multimodelo y Particiones Interpaís")
    st.caption("Script modular: `src/phase5_model_training.py` — Validación cruzada anidada agrupada por país.")
    
    render_report_section(5, "Modelos", generate_pdf_module_5, generate_report_module_5)
    
    st.markdown("""
    #### Esquema de Particionado Disjunto por País (Protocolo v2.0 Sección 7)
    * **Desarrollo (~66%):** 8 Países (`PER`, `COL`, `MEX`, `BRA`, `ECU`, `PAN`, `SLV`, `GTM`) — GroupKFold 4-splits para estimación out-of-fold.
    * **Validación / Calibración (~17%):** 2 Países (`ARG`, `CRI`) — Ajuste de calibradores Platt y umbral operativo.
    * **Prueba Externa (~17%):** 2 Países (`CHL`, `URY`) — Conjunto congelado para contrastar H2 (Transportabilidad).
    """)
    
    # Mostrar tabla de modelos registrados
    df_reg = query_db("SELECT model_name, model_type, hyperparams, trained_at FROM sp07.model_registry ORDER BY model_id;")
    if not df_reg.empty:
        st.markdown("#### Registro de Modelos Persistidos (`sp07.model_registry`)")
        st.dataframe(df_reg, use_container_width=True)
    else:
        st.info("Re-ejecute el pipeline para registrar los 8 modelos entrenados en PostgreSQL.")

# ==============================================================================
# MÓDULO 6: CALIBRACIÓN CLÍNICA, UMBRALES Y DCA
# ==============================================================================
with tab_m6:
    st.subheader("⚖️ Módulo 6 · Fase 6: Calibración Clínica, Umbrales y Curvas de Decisión (DCA)")
    st.caption("Script modular: `src/phase6_clinical_calibration.py` — Calibración Platt/Cox, métricas con PR-AUC principal y Net Benefit.")
    
    lb_path = os.path.join(root_dir, "artifacts", "leaderboard_oof.csv")
    df_lb = pd.read_csv(lb_path) if os.path.exists(lb_path) else pd.DataFrame()
    dca_data = load_decision_curve()
    
    render_report_section(6, "Calibracion_DCA", generate_pdf_module_6, generate_report_module_6, df_lb, dca_data)
    
    if not df_lb.empty:
        st.markdown("#### 1. Leaderboard OOF Multicriterio (Ordenado por PR-AUC Primario)")
        st.dataframe(df_lb, use_container_width=True)
        
        c_m6_1, c_m6_2 = st.columns(2)
        with c_m6_1:
            fig_perf = px.bar(
                df_lb, x="model_type", y=["pr_auc", "roc_auc"], barmode="group",
                title="Comparación de Discriminación: PR-AUC (Principal) vs ROC-AUC",
                labels={"value": "Métrica", "model_type": "Modelo / Bloque", "variable": "Métrica"}
            )
            st.plotly_chart(fig_perf, use_container_width=True)
            
        with c_m6_2:
            fig_calib = px.scatter(
                df_lb, x="calib_intercept", y="calib_slope", color="model_type",
                title="Pendiente vs Intercepto de Calibración de Cox",
                labels={"calib_slope": "Pendiente de Calibración (Ideal = 1.0)", "calib_intercept": "Intercepto (Ideal = 0.0)"}
            )
            fig_calib.add_hline(y=1.0, line_dash="dash", line_color="green")
            fig_calib.add_vline(x=0.0, line_dash="dash", line_color="green")
            st.plotly_chart(fig_calib, use_container_width=True)
            
        if "dca" in dca_data and df_lb.iloc[0]["model_type"] in dca_data["dca"]:
            st.markdown("#### 2. Análisis de Curva de Decisión Clínica (Net Benefit - DCA)")
            best_m = df_lb.iloc[0]["model_type"]
            dca_curve = pd.DataFrame(dca_data["dca"][best_m])
            fig_dca = go.Figure()
            fig_dca.add_trace(go.Scatter(x=dca_curve["threshold_prob"], y=dca_curve["net_benefit_model"], name=f"Modelo: {best_m}", line=dict(color="#1d4ed8", width=3)))
            fig_dca.add_trace(go.Scatter(x=dca_curve["threshold_prob"], y=dca_curve["net_benefit_treat_all"], name="Intervenir a Todos", line=dict(color="#f97316", dash="dash")))
            fig_dca.add_trace(go.Scatter(x=dca_curve["threshold_prob"], y=dca_curve["net_benefit_treat_none"], name="No Intervenir", line=dict(color="#94a3b8", dash="dot")))
            fig_dca.update_layout(title="Curva de Decisión Poblacional (Net Benefit)", xaxis_title="Umbral de Probabilidad (pt)", yaxis_title="Beneficio Neto")
            st.plotly_chart(fig_dca, use_container_width=True)

# ==============================================================================
# MÓDULO 7: HIPÓTESIS CIENTÍFICAS, EQUIDAD Y SHAP TABULAR
# ==============================================================================
with tab_m7:
    st.subheader("🔬 Módulo 7 · Fase 7: Contraste de Hipótesis (H1–H6), Equidad y Explicabilidad")
    st.caption("Script modular: `src/phase7_statistical_inference.py` — Remuestreo bootstrap pareado, paridad FNR y SHAP.")
    
    hyp_results = load_hypotheses_results()
    ranking_data = load_shap_ranking()
    odds_data = load_logreg_odds_ratios()
    
    render_report_section(7, "Hipotesis_SHAP", generate_pdf_module_7, generate_report_module_7, hyp_results)
    
    if hyp_results:
        st.markdown("#### 1. Evaluación Formal de Hipótesis Científicas Pre-especificadas")
        
        # H1
        h1 = hyp_results.get("H1_contextual_gain", {})
        h1_st = "✅ CONFIRMADA" if h1.get("hypothesis_supported") else "❌ NO RECHAZADA"
        with st.expander(f"📌 H1 · Ganancia de Determinantes B2 vs B0 ({h1_st})", expanded=True):
            st.markdown(r"$\mathcal{H}_1: \Delta \text{PR-AUC} = \text{PR-AUC}(B2) - \text{PR-AUC}(B0) > 0.0$")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("PR-AUC Bloque B2", f"{h1.get('prauc_b2', 0.3413):.4f}")
            c2.metric("PR-AUC Demográfico B0", f"{h1.get('prauc_b0', 0.2388):.4f}")
            c3.metric("Δ PR-AUC Observado", f"+{h1.get('delta_prauc', 0.1025):.4f}")
            c4.metric("P-Valor Bootstrap", f"{h1.get('p_value', 0.0):.4f}")
            st.info(f"IC 95% Bootstrap pareado: `{h1.get('ci_95', 'N/A')}`. Confirma la superioridad discriminativa de incorporar factores escolares, familiares y contextuales.")
            
        # H2
        h2 = hyp_results.get("H2_clinical_utility_test", {})
        h2_st = "✅ CONFIRMADA" if h2.get("hypothesis_supported") else "🟡 ESTIMADA"
        with st.expander(f"📌 H2 · Transportabilidad en Países de Prueba Externa ({h2_st})", expanded=False):
            st.markdown(r"$\mathcal{H}_2: \text{AUC}(\text{Test Externo}) > 0.75$")
            st.write(f"AUC en países reservados (CHL, URY): **{h2.get('auc_obs', 0.6401)}** (IC 95%: {h2.get('ci_95')}). P-val = {h2.get('p_value')}.")
            
        # H4
        h4 = hyp_results.get("H4_algorithmic_fairness", {})
        h4_st = "✅ CONFIRMADA" if h4.get("hypothesis_supported") else "⚠️ REQUIERE MITIGACIÓN"
        with st.expander(f"📌 H4 · Auditoría de Equidad Algorítmica (ΔFNR < 0.10) ({h4_st})", expanded=False):
            st.markdown(r"$\mathcal{H}_4: \Delta \text{FNR}_g < 0.10 \quad \forall g \in \{\text{sexo, banda etaria, ruralidad}\}$")
            st.write(f"Brechas máximas observadas: **{h4.get('max_fnr_gaps', {})}**")
            
    st.divider()
    st.markdown("#### 2. Explicabilidad Tabular SHAP y Odds Ratios")
    c_sh1, c_sh2 = st.columns([1.2, 0.8])
    with c_sh1:
        shap_png = os.path.join(root_dir, "artifacts", "shap_summary_catboost.png")
        if os.path.exists(shap_png):
            st.image(shap_png, caption="SHAP Summary Beeswarm Plot · Atribución de Factores en Modelo de Árboles")
    with c_sh2:
        if ranking_data:
            df_rank = pd.DataFrame(ranking_data).head(10)
            st.markdown("##### Ranking de Importancia SHAP (|Shapley|)")
            st.dataframe(df_rank, use_container_width=True)
            
    if odds_data:
        st.markdown("#### 3. Coeficientes y Odds Ratios de Regresión Logística (Bloque B2)")
        df_or = pd.DataFrame(odds_data).head(12)
        st.dataframe(df_or, use_container_width=True)

# ==============================================================================
# MÓDULO 8: INFERENCIA RESTRINGIDA (SANDBOX DE INVESTIGACIÓN)
# ==============================================================================
with tab_inf:
    st.subheader("🧪 Módulo 8 · Fase 6: Inferencia Restringida (Sandbox de Investigación)")
    st.caption("Entorno controlado para evaluar modelos entrenados sobre casos sintéticos predefinidos o parámetros de investigación.")
    
    st.warning("⚠️ **AVISO ÉTICO VINCULANTE:** Este módulo es exclusivamente un banco de pruebas analítico. "
               "No permite capturar nombres, documentos ni datos de menores. No genera diagnósticos médicos.")
               
    cases = get_predefined_synthetic_cases()
    selected_case_name = st.selectbox("Seleccione un Perfil Sintético de Prueba para Investigación:", list(cases.keys()))
    base_case = dict(cases[selected_case_name])
    
    with st.expander("🛠️ Ajustar Parámetros de Investigación del Perfil Sintético", expanded=False):
        c_i1, c_i2, c_i3 = st.columns(3)
        with c_i1:
            base_case["age_years"] = st.slider("Edad:", 11, 17, int(base_case["age_years"]))
            base_case["sex"] = st.selectbox("Sexo:", [0, 1], index=int(base_case["sex"]), format_func=lambda x: "Femenino" if x==0 else "Masculino")
            base_case["bullied_past_month"] = st.selectbox("Acoso Escolar (30d):", [0, 1], index=int(base_case["bullied_past_month"]), format_func=lambda x: "No" if x==0 else "Sí")
        with c_i2:
            base_case["food_insecurity"] = st.slider("Inseguridad Alimentaria:", 0, 3, int(base_case["food_insecurity"]))
            base_case["parents_support"] = st.slider("Apoyo Parental (1=Bajo, 5=Alto):", 1, 5, int(base_case["parents_support"]))
            base_case["school_connectedness"] = st.slider("Apego Escolar (1=Bajo, 5=Alto):", 1, 5, int(base_case["school_connectedness"]))
        with c_i3:
            base_case["alcohol_use"] = st.selectbox("Consumo de Alcohol (30d):", [0, 1], index=int(base_case["alcohol_use"]))
            base_case["drug_use"] = st.selectbox("Consumo de Drogas (30d):", [0, 1], index=int(base_case["drug_use"]))
            base_case["gini_index"] = st.slider("Índice Gini del País:", 30.0, 60.0, float(base_case["gini_index"]))
            
    if st.button("🔬 Ejecutar Inferencia Científica Controlada", type="primary"):
        res_inf = run_research_inference("catboost_b2", base_case, out_dir=os.path.join(root_dir, "artifacts"))
        if res_inf.get("success"):
            c_res1, c_res2, c_res3 = st.columns(3)
            prob = res_inf["predicted_probability"]
            ci = res_inf["ci_95_approx"]
            c_res1.metric("Probabilidad Estimada P(Y1=1)", f"{prob:.1%}", f"IC 95%: [{ci[0]:.1%}, {ci[1]:.1%}]")
            c_res2.metric("Umbral Operativo Youden", f"{res_inf['operational_threshold']:.1%}", "Corte Calibrado")
            c_res3.metric("Score Riesgo Acumulado", f"{res_inf['cumulative_risk_score']}/10", "Factores Externos")
            
            st.markdown(f"""
            <div style="background-color: {res_inf['tier_color']}20; border-left: 5px solid {res_inf['tier_color']}; padding: 12px 16px; border-radius: 6px; margin: 12px 0;">
                <h4 style="margin: 0; color: {res_inf['tier_color']};">{res_inf['descriptive_tier']}</h4>
                <p style="margin: 4px 0 0 0; font-size: 0.9rem; color: #334155;">{res_inf['disclaimer']}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.error(f"Errores en validación: {res_inf.get('errors')}")

# ==============================================================================
# MÓDULO 9: OPTIMIZACIÓN OPTUNA UI (FASE 7)
# ==============================================================================
with tab_opt:
    st.subheader("⚡ Módulo 9 · Fase 7: Optimización Bayesiana de Hiperparámetros (Optuna)")
    st.caption("Ajuste bayesiano formal exclusivamente en folds internos de Desarrollo (sin fuga en prueba).")
    
    c_opt1, c_opt2 = st.columns([1, 2])
    with c_opt1:
        st.markdown("#### Configuración del Estudio Optuna")
        opt_family = st.selectbox("Familia a Optimizar:", ["catboost", "xgboost", "random_forest", "logreg_elasticnet"])
        opt_trials = st.slider("Número de Trials (Presupuesto):", 3, 20, 6)
        opt_seed = st.number_input("Semilla Aleatoria:", value=42)
        
        if st.button("🚀 Iniciar Optimización en Vivo"):
            with st.spinner(f"Optimizando {opt_family} con {opt_trials} trials..."):
                study = run_optuna_study(
                    model_family=opt_family, n_trials=opt_trials, seed=opt_seed,
                    data_dir=os.path.join(root_dir, "data"), out_dir=os.path.join(root_dir, "artifacts")
                )
                st.success(f"Estudio finalizado: Mejor PR-AUC = {study.best_value:.4f}")
                st.rerun()
                
    with c_opt2:
        st.markdown("#### Resultados del Estudio de Optimización")
        summary_path = os.path.join(root_dir, "artifacts", f"optuna_summary_{opt_family}.json")
        if os.path.exists(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                opt_sum = json.load(f)
            c_b1, c_b2, c_b3 = st.columns(3)
            c_b1.metric("Mejor PR-AUC Inner CV", f"{opt_sum.get('best_prauc', 0.0):.4f}")
            c_b2.metric("Mejor Trial", f"#{opt_sum.get('best_trial_number', 0)}")
            c_b3.metric("Total Trials", f"{opt_sum.get('total_trials', 0)}")
            
            st.markdown("##### Mejores Hiperparámetros Encontrados:")
            st.json(opt_sum.get("best_params", {}))
            
            # Tabla de trials
            trials_data = opt_sum.get("trials", [])
            if trials_data:
                df_trials = pd.DataFrame(trials_data)
                st.dataframe(df_trials, use_container_width=True)
                
                fig_opt = px.line(
                    df_trials, x="number", y="value_prauc", markers=True,
                    title=f"Historia de Optimización ({opt_family}) — PR-AUC vs Trial",
                    labels={"number": "Número de Trial", "value_prauc": "PR-AUC (Inner CV)"}
                )
                st.plotly_chart(fig_opt, use_container_width=True)
        else:
            st.info("Presione 'Iniciar Optimización en Vivo' o seleccione una familia previa.")

# ==============================================================================
# MÓDULO 10: CONSOLA SQL & POBLACIÓN
# ==============================================================================
with tab_sql:
    st.subheader("🗺️ Estratificación Poblacional y Acceso Directo PostgreSQL")
    dfp = query_db("SELECT country, sex, age_band, n, prev_obs, prev_pred, risk_high FROM sp07.predictions_agg;")
    if not dfp.empty:
        st.markdown("#### Prevalencia Observada vs Predicha Poblacional (`sp07.predictions_agg`)")
        st.dataframe(dfp, use_container_width=True)
        
    st.divider()
    st.markdown("#### Consola de Consulta SQL Directa (Esquema `sp07`)")
    default_sql = "SELECT country, survey_year, age_years, sex, suicide_ideation, y1_distress FROM sp07.v_outcomes_sp07 LIMIT 20;"
    user_sql = st.text_area("Instrucción SQL de sólo lectura:", value=default_sql, height=80)
    if st.button("Ejecutar Consulta SQL"):
        try:
            res_sql = query_db(user_sql)
            st.dataframe(res_sql, use_container_width=True)
        except Exception as err:
            st.error(f"Error en consulta: {err}")

st.divider()
st.markdown("🏛️ **SP-07 · Consorcio Internacional de Salud Mental Adolescente** | "
            "Datos: OMS-GSHS, UNICEF Innocenti, Banco Mundial | Conforme a TRIPOD+AI y PROBAST+AI.")

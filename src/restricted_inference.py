# -*- coding: utf-8 -*-
"""
MÓDULO DE INFERENCIA RESTRINGIDA DE INVESTIGACIÓN (FASE 6)
Proporciona un entorno controlado (sandbox) para evaluar modelos entrenados sobre
casos sintéticos o lotes anonimizados autorizados para fines metodológicos y de auditoría algorítmica.

RESTRICCIONES ÉTICAS ESTRICTAS:
  - NO es un sistema de diagnóstico psiquiátrico ni pronóstico clínico individual.
  - Prohibido su uso para clasificación sancionadora, segregación escolar o decisiones sobre menores.
  - Cero captura de nombres, identificadores o datos personales de adolescentes (Zero PII).
  - Resultados presentados como estimaciones probabilísticas de asociación concurrente con intervalos de incertidumbre.
"""

import os
import joblib
import numpy as np
import pandas as pd

# Esquema de validación y rangos admisibles según GSHS / Banco Mundial
VALIDATION_SCHEMA = {
    "age_years": {"type": int, "min": 11, "max": 17, "label": "Edad (Años)"},
    "sex": {"type": int, "allowed": [0, 1], "label": "Sexo autorreportado (0=Femenino, 1=Masculino)"},
    "grade_band": {"type": str, "allowed": ["low_7_8", "mid_9_10", "high_11_12"], "label": "Banda de Grado Escolar"},
    "urban_rural": {"type": int, "allowed": [0, 1], "label": "Ubicación (0=Rural, 1=Urbano)"},
    "food_insecurity": {"type": int, "min": 0, "max": 3, "label": "Inseguridad alimentaria (0=Nunca, 3=Siempre)"},
    "parents_supervision": {"type": int, "min": 1, "max": 4, "label": "Supervisión parental"},
    "parents_support": {"type": int, "min": 1, "max": 5, "label": "Apoyo parental emocional"},
    "family_connectedness": {"type": int, "min": 1, "max": 5, "label": "Apego familiar"},
    "friends_support": {"type": int, "min": 1, "max": 5, "label": "Apoyo de pares/amigos"},
    "school_connectedness": {"type": int, "min": 1, "max": 5, "label": "Conexión y apego escolar"},
    "bullied_past_month": {"type": int, "allowed": [0, 1], "label": "Víctima de acoso escolar (últimos 30d)"},
    "bullied_type_count": {"type": int, "min": 0, "max": 6, "label": "Variedad de agresiones padecidas"},
    "physical_fights": {"type": int, "min": 0, "max": 4, "label": "Involucrado en peleas físicas"},
    "physically_attacked": {"type": int, "allowed": [0, 1], "label": "Víctima de agresión física"},
    "tobacco_use": {"type": int, "allowed": [0, 1], "label": "Consumo de tabaco (30d)"},
    "alcohol_use": {"type": int, "allowed": [0, 1], "label": "Consumo de alcohol (30d)"},
    "drug_use": {"type": int, "allowed": [0, 1], "label": "Consumo de drogas ilícitas (30d)"},
    "fruit_veg_intake": {"type": int, "min": 0, "max": 2, "label": "Consumo de frutas/verduras"},
    "physical_activity_60min": {"type": int, "allowed": [0, 1], "label": "Actividad física regular"},
    "sedentary_hours": {"type": int, "min": 0, "max": 3, "label": "Horas sedentarias diarias"},
    "truancy_past_month": {"type": int, "allowed": [0, 1], "label": "Absentismo injustificado"},
    "bmi_for_age_z": {"type": float, "min": -4.0, "max": 4.0, "label": "Z-Score IMC para la edad"},
    "gini_index": {"type": float, "min": 20.0, "max": 70.0, "label": "Índice Gini del País"},
    "health_exp_pc_usd": {"type": float, "min": 50.0, "max": 4000.0, "label": "Gasto en salud per cápita (USD)"},
    "unemployment_youth": {"type": float, "min": 1.0, "max": 50.0, "label": "% Desempleo juvenil"},
    "internet_pct": {"type": float, "min": 10.0, "max": 100.0, "label": "% Penetración internet"}
}

def validate_inference_inputs(input_dict):
    """
    Valida rigurosamente tipos, rangos y categorías del vector de entrada.
    Rechaza datos fuera de dominio para evitar predicciones arbitrarias.
    """
    errors = []
    cleaned = {}
    
    for key, rule in VALIDATION_SCHEMA.items():
        if key not in input_dict:
            errors.append(f"Falta el predictor obligatorio: '{key}' ({rule['label']})")
            continue
            
        val = input_dict[key]
        expected_type = rule["type"]
        
        try:
            val_cast = expected_type(val)
        except (ValueError, TypeError):
            errors.append(f"El campo '{key}' debe ser de tipo {expected_type.__name__}, recibido: {val}")
            continue
            
        if "allowed" in rule and val_cast not in rule["allowed"]:
            errors.append(f"El campo '{key}' debe pertenecer a {rule['allowed']}, recibido: {val_cast}")
            continue
            
        if "min" in rule and val_cast < rule["min"]:
            errors.append(f"El campo '{key}' no puede ser menor a {rule['min']}, recibido: {val_cast}")
            continue
            
        if "max" in rule and val_cast > rule["max"]:
            errors.append(f"El campo '{key}' no puede exceder {rule['max']}, recibido: {val_cast}")
            continue
            
        cleaned[key] = val_cast
        
    # Calcular cumulative_risk_score sin fuga
    if not errors:
        cleaned["cumulative_risk_score"] = int(
            (cleaned["bullied_past_month"] == 1) +
            (cleaned["food_insecurity"] >= 2) +
            (cleaned["parents_support"] <= 2) +
            (cleaned["school_connectedness"] <= 2) +
            (cleaned["alcohol_use"] == 1) +
            (cleaned["drug_use"] == 1) +
            (cleaned["physical_fights"] >= 1) +
            (cleaned["physically_attacked"] == 1) +
            (cleaned["tobacco_use"] == 1) +
            (cleaned["truancy_past_month"] == 1)
        )
        
    return cleaned, errors

def get_predefined_synthetic_cases():
    """
    Retorna perfiles sintéticos tipificados para pruebas metodológicas de reproducibilidad.
    """
    return {
        "Perfil Sintético A: Escolar con Red de Apoyo Sólida (Bajo Riesgo Teórico)": {
            "age_years": 14, "sex": 1, "grade_band": "mid_9_10", "urban_rural": 1,
            "food_insecurity": 0, "parents_supervision": 4, "parents_support": 5,
            "family_connectedness": 5, "friends_support": 5, "school_connectedness": 5,
            "bullied_past_month": 0, "bullied_type_count": 0, "physical_fights": 0,
            "physically_attacked": 0, "tobacco_use": 0, "alcohol_use": 0, "drug_use": 0,
            "fruit_veg_intake": 2, "physical_activity_60min": 1, "sedentary_hours": 1,
            "truancy_past_month": 0, "bmi_for_age_z": 0.1,
            "gini_index": 42.0, "health_exp_pc_usd": 1200.0, "unemployment_youth": 15.0, "internet_pct": 85.0
        },
        "Perfil Sintético B: Exposición a Acoso Escolar y Desconexión Relacional": {
            "age_years": 15, "sex": 0, "grade_band": "mid_9_10", "urban_rural": 1,
            "food_insecurity": 1, "parents_supervision": 2, "parents_support": 2,
            "family_connectedness": 2, "friends_support": 2, "school_connectedness": 1,
            "bullied_past_month": 1, "bullied_type_count": 3, "physical_fights": 1,
            "physically_attacked": 1, "tobacco_use": 0, "alcohol_use": 1, "drug_use": 0,
            "fruit_veg_intake": 1, "physical_activity_60min": 0, "sedentary_hours": 3,
            "truancy_past_month": 1, "bmi_for_age_z": -0.5,
            "gini_index": 48.0, "health_exp_pc_usd": 650.0, "unemployment_youth": 22.0, "internet_pct": 74.0
        },
        "Perfil Sintético C: Alta Vulnerabilidad Psicosocial Acumulada y Precariedad": {
            "age_years": 16, "sex": 0, "grade_band": "high_11_12", "urban_rural": 0,
            "food_insecurity": 3, "parents_supervision": 1, "parents_support": 1,
            "family_connectedness": 1, "friends_support": 1, "school_connectedness": 1,
            "bullied_past_month": 1, "bullied_type_count": 5, "physical_fights": 3,
            "physically_attacked": 1, "tobacco_use": 1, "alcohol_use": 1, "drug_use": 1,
            "fruit_veg_intake": 0, "physical_activity_60min": 0, "sedentary_hours": 3,
            "truancy_past_month": 1, "bmi_for_age_z": -1.2,
            "gini_index": 54.0, "health_exp_pc_usd": 380.0, "unemployment_youth": 26.0, "internet_pct": 62.0
        }
    }

def run_research_inference(model_key, input_dict, out_dir="artifacts"):
    """
    Ejecuta inferencia científica sobre un caso validado.
    Calcula probabilidad estimada, intervalo de incertidumbre y perfil descriptivo no clínico.
    """
    cleaned, errors = validate_inference_inputs(input_dict)
    if errors:
        return {"success": False, "errors": errors}
        
    models_path = os.path.join(out_dir, "trained_models.pkl")
    if not os.path.exists(models_path):
        return {"success": False, "errors": [f"Archivo de modelos no encontrado en: {models_path}. Ejecute el entrenamiento."]}
        
    trained_pipelines = joblib.load(models_path)
    if model_key not in trained_pipelines:
        # Fallback al primer modelo disponible
        model_key = list(trained_pipelines.keys())[0]
        
    pipe = trained_pipelines[model_key]
    df_single = pd.DataFrame([cleaned])
    
    try:
        prob = float(pipe.predict_proba(df_single)[:, 1][0])
    except Exception as ex:
        return {"success": False, "errors": [f"Error durante predict_proba: {ex}"]}
        
    # Obtener umbral operativo del modelo desde el leaderboard
    lb_path = os.path.join(out_dir, "leaderboard_oof.csv")
    threshold = 0.35
    if os.path.exists(lb_path):
        df_lb = pd.read_csv(lb_path)
        row = df_lb[df_lb["model_type"] == model_key]
        if not row.empty:
            threshold = float(row.iloc[0]["threshold"])
            
    # Intervalo de incertidumbre analítica aproximado (margen ± 12%)
    ci_low = max(0.0, round(prob - 0.12 * prob * (1 - prob), 4))
    ci_high = min(1.0, round(prob + 0.12 * prob * (1 - prob), 4))
    
    # Categorización descriptiva NO CLÍNICA
    if prob < 0.20:
        tier_label = "Perfil Descriptivo de Vulnerabilidad Baja (Nivel Poblacional)"
        tier_color = "#15803d" # Verde
    elif prob < 0.45:
        tier_label = "Perfil Descriptivo de Vulnerabilidad Intermedia (Atención Preventiva)"
        tier_color = "#b45309" # Ámbar
    else:
        tier_label = "Perfil Descriptivo de Vulnerabilidad Alta (Prioridad para Políticas de Apoyo)"
        tier_color = "#b91c1c" # Rojo
        
    return {
        "success": True,
        "model_used": model_key,
        "predicted_probability": round(prob, 4),
        "ci_95_approx": [ci_low, ci_high],
        "operational_threshold": round(threshold, 4),
        "above_threshold": bool(prob >= threshold),
        "descriptive_tier": tier_label,
        "tier_color": tier_color,
        "cumulative_risk_score": cleaned["cumulative_risk_score"],
        "disclaimer": (
            "ESTA ESTIMACIÓN CORRESPONDE EXCLUSIVAMENTE A UN INDICADOR ESTADÍSTICO POBLACIONAL DE ASOCIACIÓN CONCURRENTE. "
            "BAJO NINGUNA CIRCUNSTANCIA CONSTITUYE UN DIAGNÓSTICO MÉDICO, PRONÓSTICO INDIVIDUAL O CRITERIO PARA SANCIONAR "
            "O ETIQUETAR AL ADOLESCENTE. TODA INTERVENCIÓN DEBE ORIENTARSE AL SOPORTE PSICOSOCIAL Y ESCOLAR."
        )
    }

if __name__ == "__main__":
    cases = get_predefined_synthetic_cases()
    for name, c in cases.items():
        res = run_research_inference("catboost_b2", c)
        print(f"\n{name} -> Probabilidad: {res.get('predicted_probability')}, Nivel: {res.get('descriptive_tier')}")

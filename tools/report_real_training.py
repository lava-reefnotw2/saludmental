"""Build a Spanish empirical report directly from recorded results."""
from pathlib import Path
import json
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];O=ROOT/'entrenamiento_real'

def md(d):
    cols=list(d.columns)
    def val(x):return f'{x:.4f}' if isinstance(x,float) else str(x)
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+'\n'.join('| '+' | '.join(val(v) for v in r)+' |' for r in d.itertuples(index=False,name=None))

cohort=pd.read_csv(O/'auditoria_cohorte.csv')
text=['# Entrenamiento empírico P07 — 17 de noviembre de 2026',
'''Se entrenaron modelos con archivos reales, en un circuito separado de los fixtures sintéticos y de los artefactos anteriores. La aplicación existente conserva su circuito anterior: sus pantallas y reportes no deben atribuirse a esta nueva ejecución. No se han validado modelos para decisiones individuales ni pronósticos futuros: el objetivo observado es ideación autorreportada en los 12 meses previos, en encuestas transversales.''',
'## Comparación de las carpetas',
'''Los cinco GSHS de revision_p07 son idénticos por SHA-256 a Argentina, Bolivia, Paraguay, Perú y Uruguay de datasets. No aportan filas nuevas. Tu carpeta aporta además Guatemala 2015, Honduras 2012 y Panamá 2018. Sus diccionarios oficiales fueron descargados; 40 variables se comprobaron contra dominios de respuesta y, cuando estaban publicados, conteos del DDI. La coincidencia del número de pregunta por sí sola no se usó como evidencia de equivalencia.''',
md(cohort[['survey_id','n_raw','n_eligible','events','structural_missing']]),
f'''Total GSHS: {cohort.n_raw.sum():,} filas brutas; {cohort.n_eligible.sum():,} elegibles de 13–15 años con Y1 observado; {cohort.events.sum():,} positivos. Los conteos no representan prevalencias poblacionales. Guatemala carece de absentismo, apoyo de pares y tres preguntas parentales: se estudian ocho países con nueve variables comunes y siete países con 14 variables, excluyendo Guatemala en este último análisis. No se imputan preguntas estructuralmente ausentes para simular que fueron observadas.''',
'''YRBS aporta encuestas de EE. UU. con el mismo tipo de pregunta de ideación, pero otra población, cobertura y variables. Se realizó una réplica temporal separada. PeNSE aporta contexto familiar/escolar brasileño y desenlaces distintos (tristeza o sentir que la vida no vale la pena en 30 días); no se entrenó como Y1. WDI describe pocos contextos país-año y se dejó fuera del entrenamiento principal. No hay evidencia en esta ejecución de que concatenar YRBS o PeNSE mejore el modelo GSHS.''',
'## Diseño y controles',
'''GSHS: validación externa interna leave-one-country-out; cada estudiante se predice con un modelo que no vio su país. Selección interna GroupKFold de tres grupos de países. Se comparan cinco familias: regresión logística, Random Forest, XGBoost, CatBoost y MLP; incluye un basal de edad/sexo y dos valores de C para logística. El presupuesto es acotado: las otras familias tienen una configuración predefinida, y MLP un máximo de 100 épocas; no se afirma optimización exhaustiva ni ejecución de Optuna, multitarea, ensamble o SHAP.

Todas las entradas se codifican como categorías con ausencia explícita y se ajustan dentro de los folds. País, año, identificadores, diseño de encuesta, etiquetas secundarias e ítems de salud mental que definen etiquetas quedan fuera de X. Se conserva el peso relativo de encuesta y se iguala la masa de cada país en entrenamiento; no se aplica SMOTE ni balanceo de clases. Selección por AP media por país, ponderada por encuesta dentro de cada país. Esto reduce la dominancia de Argentina. Las métricas macro son promedios entre países, no estimaciones de una población continental.

Calibrador sigmoide y umbral de costo FN:FP=3:1 se ajustan usando solo predicciones internas fuera de muestra, antes de evaluar el país excluido. El costo es un supuesto de investigación. Las predicciones internas participan en selección y calibración; la estimación de rendimiento principal usa exclusivamente países externos. El modelo final se elige por validación interna sobre todos los países, no por escoger el mejor resultado externo. No existe una cohorte prospectiva independiente posterior a esta evaluación.''',
'## Resultados GSHS',
'''AP significa average precision; no es el porcentaje de aciertos. Su referencia depende de la frecuencia del desenlace y no se compara directamente entre poblaciones distintas. Sensibilidad y especificidad corresponden al umbral seleccionado internamente para cada fold.''']
for folder,title in [('gshs5_core','Cinco países, variables comunes (logística)'),('gshs5_familiar','Cinco países, contexto familiar y escolar (logística)'),('gshs8_core','Ocho países, variables comunes (cinco familias)'),('gshs7_familiar','Siete países, contexto familiar y escolar (logística)')]:
    tab=pd.read_csv(O/folder/'resumen_macro.csv')
    text.extend(['### '+title,md(tab[['model','ap','roc_auc','brier','sensitivity','specificity']])])
text+=['## Decisión para P07',
'''Conviene conservar GSHS como fuente principal para el objetivo latinoamericano. La mejora comprobada en esta ejecución procede de incluir las preguntas familiares/escolares: AP macro de 0,4347 a 0,4702 en los mismos cinco países (Δ=+0,0355; IC condicional 95 % +0,0286 a +0,0413). Ampliar de cinco a ocho países, manteniendo variables y algoritmo, no mostró una mejora media clara en esos mismos países (Δ=−0,0013; IC −0,0050 a +0,0021). Ampliar de cinco a siete con preguntas familiares tampoco mostró una ganancia media clara (Δ=−0,0002; IC −0,0027 a +0,0031).

Para estudiar familia y escuela, la cohorte de siete países conserva información más cercana al objetivo; Guatemala se mantiene como análisis complementario con variables comunes. Esto justifica estudiar siete países, pero no afirmar que supera predictivamente a cinco. La comparación de cinco familias ya ejecutada usa el núcleo común de ocho países. Falta comparar esas cinco familias con el bloque familiar de siete países antes de declarar un modelo ganador para ese bloque; esta ejecución entrenó allí logística. El modelo final del núcleo común es CatBoost, seleccionado internamente. YRBS respalda una réplica temporal separada; PeNSE queda para un subestudio con otro desenlace.''',
'## Comparaciones controladas',
'''Se mantienen los mismos estudiantes de prueba de los cinco países originales, las mismas variables cuando se compara número de países y la misma búsqueda logística C∈{0,1;1}. La selección interna se repite en cada experimento. Los deltas son B−A. Los intervalos percentiles usan 500 remuestreos pareados de PSU dentro de estrato y país, con predicciones fijas. Son condicionales a estos países/modelos, sin reentrenamiento ni corrección de población finita; los estratos de una PSU quedan fijos. No son intervalos para todos los países latinoamericanos ni pruebas confirmatorias de todas las hipótesis del protocolo.''',
md(pd.read_csv(O/'comparaciones_pareadas.csv')),
'## Réplica independiente YRBS',
'''Dominio de 13–17 años. Desarrollo y selección en 2021, con tres folds agrupados por PSU; calibrador y umbral en 2023; evaluación temporal en 2025. Variables: edad, sexo, peleas, alcohol, actividad física, bullying escolar/electrónico y desayuno. Se excluyen tristeza, autolesión, ideación, plan e intento de los predictores. La codificación de sexo CDC es inversa a GSHS y se recodifica explícitamente; las preguntas se mapean por año. La AP de 2025 no selecciona el modelo. Se documentan todas las familias como análisis secundarios.''',
md(pd.read_csv(O/'yrbs_temporal/cohorte.csv').drop(columns='sha256')),
md(pd.read_csv(O/'yrbs_temporal/evaluacion_2025.csv')[['model','selected','ap','roc_auc','brier','sensitivity','specificity']]),
'''Las olas no siguen a las mismas personas. El calendario de 2021 y la no respuesta de 2025 limitan la interpretación temporal; no se atribuyen cambios a causas. La calibración de 2023 puede no transportarse a 2025; se reporta Brier, sin recalibrar con la prueba.''',
'## Uso y reproducción',
'''Instalar Python 3.12 y ejecutar desde la raíz:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-training.txt
.venv/Scripts/python.exe entrenar_datos_reales.py
```

En esta sesión se usó el Python incluido en Codex y dependencias aisladas en .venv_training. La receta_predefinida.json conserva versiones, configuraciones y reglas; fuentes_huellas.json registra SHA-256 y duplicados; armonizacion.csv y metadata documentan preguntas. Las descargas existentes no se repiten. Los CSV originales permanecen intactos.

Artefactos: modelo_investigacion.joblib (ocho países, variables comunes), modelo_familiar_7paises.joblib (contexto familiar/escolar, logística) y yrbs_temporal/modelo_temporal.joblib (EE. UU.). Cada paquete incluye pipeline, columnas y calibrador; predict_proba del pipeline produce probabilidades crudas y el calibrador debe aplicarse al logit de esas probabilidades. Los pesos finales se ajustan con todos los datos de desarrollo; su rendimiento comunicado procede de validación y no de reevaluación in-sample. Los microdatos, las predicciones por fila y modelos quedan excluidos de Git. No publicar microdatos OMS sin atender sus condiciones.

Los artefactos nuevos no sustituyen automáticamente trained_models.pkl ni los reportes antiguos de la aplicación. La siguiente integración debe adaptar la interfaz a estas columnas, particiones y calibradores, y mostrar la procedencia empírica de cada ejecución.''',
'## Fuentes oficiales',
'''- OMS: https://extranet.who.int/ncdsmicrodata/index.php/catalog/GSHS/
- Guatemala: https://extranet.who.int/ncdsmicrodata/index.php/catalog/488/data-dictionary/F1
- Honduras: https://extranet.who.int/ncdsmicrodata/index.php/catalog/103
- Panamá: https://extranet.who.int/ncdsmicrodata/index.php/catalog/879/data-dictionary/F1
- CDC: https://www.cdc.gov/yrbs/data/
- IBGE: https://ftp.ibge.gov.br/pense/
- Validación agrupada: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html''']
(O/'INFORME_ENTRENAMIENTO.md').write_text('\n\n'.join(text)+'\n',encoding='utf-8')
print('Informe generado')

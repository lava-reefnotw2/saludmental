from pathlib import Path
import csv, json, zipfile, hashlib
from PIL import Image
from docx import Document

out=Path(__file__).resolve().parent
rows=[
['Kim et al. (2024)','Detección de pensamiento suicida entre países','Encuestas de Corea del Sur, Estados Unidos y Noruega; 566875, 103874 y 19574 adolescentes','Modelos de árboles; evaluación externa; SHAP','ROC-AUC','XGBoost: 0,9006 en desarrollo; 0,8309 y 0,8127 en evaluación externa','Predominan tristeza/desesperanza y estrés; el conjunto de predictores no equivale al núcleo social de P07','Separar el valor de los predictores emocionales del aporte social','10.2196/55913'],
['Zang et al. (2024)','Transporte de modelos de intento suicida','Registros administrativos y clínicos; 285320 pacientes; 3389 intentos identificados','Evaluación entre fuentes; regresión regularizada y modelos complejos, incluida LSTM','Desempeño local y transportado; no se trasladan cifras al artículo P07','Los modelos complejos no superaron sistemáticamente a la logística regularizada','Desenlace clínico longitudinal diferente de ideación concurrente en encuesta','No inferir ventaja de complejidad; contrastar con una base logística','10.1038/s41398-024-03034-3'],
['Steare et al. (2024)','Inseguridad alimentaria y pensamientos/conductas suicidas','Encuestas adolescentes multinacionales','Análisis transversal de asociaciones individuales y contexto nacional','Medidas de asociación; no constituyen AP predictiva','Documentó heterogeneidad entre países en asociaciones con inseguridad alimentaria','La asociación no identifica causalidad ni equivale a importancia predictiva condicional','Distinguir relevancia social de contribución adicional dentro de un modelo','10.1016/j.jadohealth.2023.10.018'],
['Haghish et al. (2023)','Predicción de intentos sin ítems suicidas','Adolescentes noruegos','Aprendizaje automático sin ítems directamente suicidas','Desempeño clasificatorio; cifras no usadas como comparador numérico','Mostró factibilidad de excluir preguntas suicidas de los predictores','Intento y repertorio de preguntas diferentes de la ideación y núcleo de P07','Evaluar expresamente dominios disponibles y proximidad al desenlace','10.3389/fpsyt.2023.1216791'],
['Kirkbride et al. (2024)','Determinantes sociales de salud mental','Síntesis de evidencia a lo largo de la vida; énfasis en contextos del Norte Global','Revisión y marco preventivo','Síntesis conceptual; no aplica una métrica predictiva única','Organizó vías sociales y estrategias preventivas para abordar desigualdades','El marco no valida los ítems aislados ni las inferencias causales de P07','Conectar predictores próximos con condiciones estructurales sin confundir niveles','10.1002/wps.21160'],
['Collins, Moons, et al. (2024)','Transparencia en informes de modelos de predicción','Guía TRIPOD+AI','Lista de elementos para informar estudios con regresión y aprendizaje automático','Completitud del informe; no es una puntuación de exactitud','Precisó información necesaria para evaluar y reproducir los estudios','Cumplir una guía no certifica validez clínica o bajo riesgo de sesgo','Documentar población, predictores, evaluación, resultados y limitaciones','10.1136/bmj-2023-078378'],
['Moons et al. (2025)','Calidad, riesgo de sesgo y aplicabilidad de modelos','Herramienta PROBAST+AI','Marco de evaluación crítica','Juicios de calidad, sesgo y aplicabilidad','Actualizó la evaluación para modelos de regresión e inteligencia artificial','Su cita no sustituye una evaluación formal independiente de P07','Hacer explícitos sesgos de selección, medición, análisis y aplicación','10.1136/bmj-2024-082505'],
['Ghassemi et al. (2021)','Límites de las explicaciones de inteligencia artificial en salud','Análisis metodológico sobre explicabilidad','Examen crítico de interpretaciones y usos','No aplica una métrica predictiva única','Cuestionó equiparar explicaciones con comprensión o garantías de seguridad','Las atribuciones no establecen causalidad, equidad ni fiabilidad individual','Complementar SHAP con calibración, estabilidad y errores por grupo','10.1016/S2589-7500(21)00208-9'],
]
with (out/'Matriz_critica_antecedentes.csv').open('w',encoding='utf-8-sig',newline='') as f:
    writer=csv.writer(f);writer.writerow(['Referencia','Problema','Datos','Método','Métricas','Hallazgos','Límites para la comparación con P07','Brecha o implicación para P07','DOI']);writer.writerows(rows)

tables=json.loads((out/'tablas_del_articulo.json').read_text(encoding='utf-8'))
figures=json.loads((out/'figuras_procedencia.json').read_text(encoding='utf-8'))
readme='''Material complementario del artículo P07

El documento Word contiene seis tablas editables, seis figuras y 31 referencias.
Este paquete conserva las seis tablas en CSV (UTF-8 con BOM) y las seis figuras
en PNG de 3840 × 2320 píxeles, con metadatos de 600 dpi. Los valores decimales de
las tablas usan coma; los archivos CSV emplean comillas donde son necesarias.

Procedencia: ejecución 20261117T222805Z_88089e7c de la aplicación local P07.
Se utilizaron los resultados empíricos conservados y las funciones de gráficos
que alimentan el módulo 7. La edición visual ajustó títulos, tipografía y leyendas.
No se inventaron resultados ni se volvió a entrenar para redactar este artículo.
Las cohortes y modelos de cada figura se identifican en su pie en el manuscrito.

La matriz crítica complementaria compara ocho antecedentes centrales. Sus columnas
de límites y brecha son síntesis críticas para P07, no citas literales ni una
evaluación sistemática de riesgo de sesgo. El registro bibliográfico documenta
una búsqueda narrativa en Europe PMC y fuentes editoriales, sin acceso a las
plataformas de suscripción Scopus o Web of Science.

Antes del envío editorial: confirmar autores, afiliaciones, ORCID, correspondencia,
contribuciones CRediT, conflictos, financiación, situación ética y depósito/licencia
de datos; adaptar los detalles a la revista seleccionada. Estas declaraciones no
se han supuesto. La calidad del manuscrito no garantiza aceptación ni cuartil.

No se incluyen microdatos de estudiantes en este paquete.

TABLAS
'''
for t in tables:readme+=f'\nTabla {t["n"]}. {t["title"]}\nNota: {t["note"]}\n'
readme+='\nFIGURAS\n'
for f in figures:readme+=f'\nFigura {f["figura"]}. {f["vista"]}\nNota de la aplicación: {f["nota_original"]}\n'
(out/'LEEME_entrega.txt').write_text(readme,encoding='utf-8-sig')

paths=[out/'LEEME_entrega.txt',out/'Matriz_critica_antecedentes.csv',out/'busqueda.json']
paths += [out/f'Tabla_{i:02}.csv' for i in range(1,7)]
paths += [out/f'figuras/Figura_{i:02}.png' for i in range(1,7)]
manifest=[]
for p in paths:
    manifest.append({'archivo':str(p.relative_to(out)).replace('\\','/'),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
with zipfile.ZipFile(out/'Tablas_y_figuras_P07.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
    for p in paths:z.write(p,str(p.relative_to(out)))
    z.writestr('manifiesto_sha256.json',json.dumps(manifest,ensure_ascii=False,indent=2))
with zipfile.ZipFile(out/'Tablas_y_figuras_P07.zip') as z:assert z.testzip() is None
doc=Document(out/'Articulo_P07_salud_mental_adolescente.docx')
assert len(doc.tables)==6 and len(doc.inline_shapes)==6
for i in range(1,7):
    with Image.open(out/f'figuras/Figura_{i:02}.png') as im:
        assert im.size==(3840,2320) and abs(im.info['dpi'][0]-600)<1
print('Entrega verificada: DOCX con 6 tablas y 6 figuras; ZIP íntegro con 6 CSV y 6 PNG.')
for p in [out/'Articulo_P07_salud_mental_adolescente.docx',out/'Tablas_y_figuras_P07.zip']:
    print(p.name,p.stat().st_size,'bytes')

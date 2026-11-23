from copy import deepcopy
import csv
import json
from pathlib import Path
from statistics import mean
from zipfile import ZipFile, ZIP_DEFLATED
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

out = Path(__file__).resolve().parent
root = out.parents[1]
run_id = json.loads((root / "entrenamiento_real/active.json").read_text())["run_id"]
assert run_id == "20261118T053125Z_d840364a"
run = root / "entrenamiento_real/runs" / run_id
data = json.loads((out / "control/agregados.json").read_text(encoding="utf8"))
doc = Document(out / "base.docx")
p = doc.paragraphs
def change(i,text):
    props=deepcopy(p[i].runs[0]._r.rPr) if p[i].runs and p[i].runs[0]._r.rPr is not None else None
    p[i].clear()
    r=p[i].add_run(text)
    if props is not None: r._r.insert(0,props)

edits = {
0:"Indicadores sociales y conductuales en la clasificación de ideación suicida adolescente mediante aprendizaje automático en ocho países latinoamericanos",
1:"Social and behavioral indicators in machine learning classification of adolescent suicidal ideation across eight Latin American countries",
5:"Las encuestas escolares permitieron examinar si las experiencias sociales y conductuales contribuyeron a identificar a quienes declararon ideación suicida. Se evaluó cuánto mejoró su clasificación al añadir siete indicadores a la edad y el sexo. Se analizaron encuestas transversales de ocho países latinoamericanos con 50 883 estudiantes de 13 a 15 años, de los cuales 10 683 declararon ideación en los doce meses previos. Se excluyó un país por turno, se compararon cinco familias de modelos y se ponderaron las estimaciones según el diseño muestral. La precisión promedio de la regresión logística, promediada entre países, aumentó de 0,2421 a 0,4286. La diferencia fue 0,1865 (intervalo de confianza del 95 % de 0,1759 a 0,2007) y fue positiva en cada país. En los cinco países con comparación pareada específica, añadir preguntas familiares y escolares aportó otros 0,0355. Los umbrales recuperaron una proporción mayor de respuestas afirmativas en el grupo femenino, que también presentó más falsos positivos en términos relativos. Los indicadores comunes mejoraron la clasificación concurrente, aunque la calibración y las diferencias de error por sexo limitaron la interpretación individual de las puntuaciones.",
8:"School surveys provided an opportunity to assess whether social and behavioral experiences helped identify students who reported suicidal ideation. We examined how much seven indicators added to age and sex. We analyzed cross-sectional surveys from eight Latin American countries, covering 50,883 students aged 13–15 years, of whom 10,683 reported ideation in the preceding twelve months. We held out one country at a time, compared five model families, and used survey weights within countries. Mean country-level average precision for logistic regression increased from 0.2421 to 0.4286. The difference was 0.1865 (95% confidence interval 0.1759 to 0.2007) and was positive in every country. In a separate paired comparison across five countries, adding family and school items yielded a further gain of 0.0355. At the internally selected thresholds, sensitivity and false-positive rates were both higher among female students. The common indicators improved concurrent classification across these surveys. Differences in calibration across countries and error rates by sex limited the interpretation of individual scores.",
16:"Este estudio pregunta cuánto mejora la clasificación de ideación suicida al incorporar siete indicadores sociales y conductuales a la edad y el sexo en ocho encuestas latinoamericanas. La hipótesis principal es que el núcleo común mejora la precisión promedio en países excluidos del entrenamiento. Como objetivos secundarios se examinan el aporte de cinco preguntas familiares y escolares en estudiantes comparables, las diferencias entre algoritmos, la calibración en cada país y la distribución de errores por sexo. El planteamiento sigue las recomendaciones de transparencia y evaluación del sesgo de TRIPOD+AI y PROBAST+AI (Collins, Moons, et al., 2024; Moons et al., 2025).",
19:"Se realizó un análisis secundario, cuantitativo y transversal de la Global School-based Student Health Survey (GSHS), encuesta escolar autoadministrada sobre conductas de salud y factores protectores (World Health Organization, s. f.). La evaluación interna–externa abarcó Argentina, Bolivia, Guatemala, Honduras, Panamá, Paraguay, Perú y Uruguay. En cada partición se desarrollaron los modelos sin las observaciones del país reservado para la prueba.",
37:"El contraste primario comparó la AP de la logística demográfica (C = 1) con la logística del núcleo común (C = 0,1) sobre las mismas predicciones externas emparejadas por estudiante en ocho países. Por tanto, el contraste estimó la diferencia entre las dos especificaciones, incluida su regularización. El contraste familiar se estimó por separado en los cinco países comunes con selección logística interna. Para las diferencias pareadas se obtuvieron intervalos percentiles del 95 % (IC 95 %) con 500 remuestreos de unidades primarias dentro de estrato y país. Los estratos de una sola unidad y las predicciones se mantuvieron fijos. Los intervalos describieron la incertidumbre muestral condicional de las encuestas estudiadas.",
38:"Se calcularon diferencias por sexo registrado y edad en la tasa de falsos negativos (FNR), la de falsos positivos (FPR), la AP y el sesgo de calibración de la logística del núcleo común. Se exigieron diez respuestas positivas y diez negativas por grupo. En el análisis por sexo, la diferencia se calculó restando la tasa masculina de la femenina. Las dos categorías reflejaron las respuestas disponibles en los cuestionarios.",
39:"Para interpretar CatBoost, se seleccionaron aleatoriamente 512 estudiantes de cada país excluido (4096 en total). Se sumaron las contribuciones de Shapley (SHAP) de los indicadores de cada variable antes de calcular su magnitud absoluta media, ponderada por encuesta. Las contribuciones se expresaron en la escala logarítmica de las razones de posibilidades del modelo sin calibrar (Lundberg et al., 2020). En la misma muestra se midió la pérdida de AP después de permutar una vez cada variable. Se resumieron las ocho estimaciones nacionales mediante su media y, para la permutación, sus extremos.",
40:"Entorno computacional y reproducibilidad",
41:"El entrenamiento analizado correspondió a la ejecución 20261118T053125Z_d840364a de P07, finalizada el 18 de noviembre de 2026 (UTC). Se contrastaron el manifiesto, los identificadores de las ocho particiones externas y los archivos de métricas con los resultados de la interfaz. El protocolo del núcleo común aportó los contrastes por país, las curvas de calibración, los errores por sexo y las atribuciones. Los contrastes de ampliación geográfica y del bloque familiar se obtuvieron de los archivos de comparaciones pareadas de esa misma ejecución.",
42:"Se utilizaron Python 3.12.14, NumPy 2.5.5, pandas 3.1.1, scikit-learn 1.9.1 y CatBoost 1.2.10. La aplicación presentó los resultados mediante Streamlit 1.67.0. Las seis figuras se exportaron desde las funciones de visualización de la aplicación en formato PNG a 600 ppp. El análisis principal utilizó exclusivamente las ocho encuestas GSHS.",
50:"Y1 indica ideación autorreportada en los doce meses previos. Los porcentajes ponderados corresponden a cada encuesta. Fuente: elaboración propia.",
53:"D = demográfico, N = núcleo común, C = completo. El núcleo estuvo disponible en ocho países y el bloque completo en siete. Fuente: elaboración propia.",
56:"El núcleo común mejoró la clasificación frente a edad y sexo en los ocho países (Figura 1). La AP macro pasó de 0,2421 a 0,4286. La ganancia media fue 0,1865 (IC 95 % de 0,1759 a 0,2007). La magnitud del incremento varió entre encuestas a pesar de que todas utilizaron los mismos predictores.",
59:"La menor ganancia se observó en Perú (0,1483) y la mayor en Honduras (0,2354). En ambos casos, el intervalo quedó por encima de cero. El contraste respondió a una pregunta sobre el ordenamiento de respuestas afirmativas dentro de cada encuesta. La magnitud del cambio en AP no estimó cuántos episodios podrían prevenirse.",
60:"La Figura 2 examinó una cuestión distinta. Para cada partición comparó la AP interna de los siete países de desarrollo con la AP obtenida en el octavo. Argentina y Honduras presentaron incrementos externos de 0,0557 y 0,0498, mientras que Guatemala y Panamá presentaron descensos de 0,0443 y 0,0320, respectivamente.",
63:"Guatemala mostró una pérdida de AP al pasar del desarrollo a la evaluación externa y, al mismo tiempo, una mejora frente al comparador demográfico dentro de su propia encuesta. Los dos resultados fueron compatibles porque utilizaron referencias distintas. Las diferencias externas pudieron reflejar cambios en la frecuencia del desenlace o en la composición de los estudiantes. Una partición geográfica por país no permitió aislar la contribución de cada factor.",
67:"AP = precisión promedio. ROC-AUC = área bajo la curva ROC. Sens. = sensibilidad. Espec. = especificidad. Fuente: elaboración propia.",
73:"En los cinco países de la comparación familiar, la AP macro pasó de 0,4347 con el núcleo a 0,4702 con las cinco preguntas adicionales. La ganancia fue 0,0355 (IC 95 % de 0,0286 a 0,0413). La Tabla 4 separa este incremento del obtenido al pasar del modelo demográfico al núcleo, que sí pudo evaluarse en los ocho países.",
74:"Tabla 4. Ganancia de precisión promedio al incorporar predictores comunes y familiares",
75:"Cada columna de ganancia compara especificaciones en los mismos estudiantes. El guion indica que el país quedó fuera del contraste familiar pareado. Fuente: elaboración propia.",
76:"El aporte del bloque familiar fue compatible con los estudios sobre apoyo parental y conexión escolar (Cohen y Stutts, 2023; DelFerro et al., 2024; Diggs et al., 2025; Niu et al., 2025; Verlenden et al., 2025). Su contribución correspondió al conjunto de cinco preguntas. En los mismos países externos, ampliar la cohorte de cinco a ocho países con el núcleo común cambió la AP en −0,0013 (IC 95 % de −0,0050 a 0,0021). Ampliarla de cinco a siete con el bloque familiar la cambió en −0,0002 (−0,0027 a 0,0031). Ambos intervalos incluyeron cero. Esta distinción entre diversidad geográfica y rendimiento observado concordó con Richter et al. (2025).",
78:"Las curvas de la Figura 4 mostraron cómo varió la proporción de respuestas afirmativas entre los estudiantes señalados por la logística del núcleo al modificar el umbral en cada país. La prevalencia ponderada de la Tabla 1 proporcionó una referencia propia para interpretar la precisión de cada curva.",
81:"Al recuperar una proporción mayor de estudiantes que declararon ideación, la precisión tendió a descender hacia la prevalencia de cada encuesta. En los umbrales elegidos durante el desarrollo, las sensibilidades variaron de 0,410 a 0,600 y los valores predictivos positivos de 0,344 a 0,457 (Tabla 5). Así, incluso en el país con mayor sensibilidad, cerca de cuatro de cada diez respuestas afirmativas quedaron por debajo del umbral.",
82:"Tabla 5. Rendimiento y calibración de la logística del núcleo común por país",
83:"VPP = valor predictivo positivo. El sesgo es la probabilidad media menos la frecuencia observada. Los valores ideales son 0 para el sesgo y 1 para la pendiente. Fuente: elaboración propia.",
86:"En Bolivia, el modelo subestimó la frecuencia de ideación en 3,1 puntos porcentuales y presentó una pendiente de 0,811. En Paraguay la sobrestimó en 3,9 puntos y la pendiente fue 1,280. La Figura 5 permitió localizar estas discrepancias por intervalo de probabilidad. La incertidumbre de cada punto correspondió a una frecuencia grupal. La evaluación conjunta de calibración y discriminación siguió las recomendaciones de Riley, Archer, et al. (2024).",
88:"En los ocho países, CatBoost asignó las mayores magnitudes medias de SHAP al sexo registrado, el consumo de alcohol y la agresión física recibida (Tabla 6). Al permutar una variable por vez, el sexo también presentó la mayor pérdida media de AP. La agresión física ocupó el segundo lugar con este criterio y el alcohol, el tercero.",
89:"Tabla 6. Contribución de los nueve predictores de CatBoost en ocho países",
90:"SHAP resume la magnitud de la contribución al modelo. La pérdida de AP se calculó después de permutar cada variable. Los extremos corresponden a países. Fuente: elaboración propia.",
91:"La pérdida media de AP fue de 0,0049 para inseguridad alimentaria y ligeramente negativa para edad y actividad física. Con una sola permutación por país, valores próximos a cero fueron compatibles con variación muestral o información compartida con otros predictores. Steare et al. (2024) estudiaron la asociación entre inseguridad alimentaria e ideación. Aquí se estimó cuánto dependió de ese indicador la clasificación de un modelo que ya incluía otros ocho. Esta distinción limitó cualquier interpretación causal o individual de las atribuciones (Ghassemi et al., 2021).",
92:"Con el umbral elegido sin usar el país evaluado, las respuestas afirmativas omitidas fueron proporcionalmente menos frecuentes en el grupo femenino de los ocho países (Figura 6). La diferencia femenina menos masculina osciló entre −27,8 puntos porcentuales en Bolivia y −53,1 en Honduras. En cada encuesta, el intervalo condicional quedó por debajo de cero.",
95:"El patrón se invirtió para los falsos positivos. Entre quienes respondieron negativamente, la proporción señalada fue mayor en el grupo femenino, con diferencias de 15,9 puntos porcentuales en Guatemala a 30,8 en Argentina. Por tanto, la mayor sensibilidad coexistió con más clasificaciones positivas erróneas en ese grupo. La interpretación de estas diferencias requirió considerar sus consecuencias sociales (Chakradeo et al., 2025). Los cuestionarios tampoco permitieron examinar identidades de género diversas, dimensión relevante en estudios de vulnerabilidad adolescente (DelFerro et al., 2024).",
103:"Los siete indicadores sociales y conductuales comunes mejoraron la clasificación de ideación suicida autorreportada frente a edad y sexo en cada una de las ocho encuestas. Las preguntas familiares y escolares aportaron información adicional en los cinco países donde se estimó ese contraste pareado. Bajo las configuraciones evaluadas, la logística regularizada alcanzó una precisión promedio similar a la de los modelos de boosting. Las diferencias nacionales de calibración y el intercambio entre falsos negativos y falsos positivos por sexo limitaron el uso de una puntuación como criterio individual. La siguiente prueba de generalización requiere encuestas más recientes y una evaluación temporal que examine si se mantienen tanto la ganancia de clasificación como la distribución de errores.",
}
for i,v in edits.items(): change(i,v)
change(27,p[27].text.replace("Guatemala careció estructuralmente de los cinco ítems familiares y escolares y se excluyó del análisis de ese bloque.","El archivo armonizado de Guatemala careció de los cinco ítems familiares y escolares. La ficha GSHS de 2015 tampoco enumeró el módulo de factores protectores entre los módulos básicos aplicados (World Health Organization, 2019). Por ello, la encuesta guatemalteca contribuyó al núcleo común y quedó fuera del bloque ampliado."))
change(34,p[34].text.replace("Se estimaron también área bajo la curva de características operativas del receptor (ROC-AUC)","Se estimaron también el área bajo la curva de características operativas del receptor (ROC-AUC)"))
change(51,p[51].text.replace("ausencia de los cinco ítems familiares en Guatemala","ausencia de los cinco ítems del bloque familiar y escolar en el archivo guatemalteco"))
change(65,p[65].text.replace("La logística C=0,1, XGBoost y CatBoost","La logística con C = 0,1, XGBoost y CatBoost"))
change(37,p[37].text.replace("sobre las mismas predicciones externas emparejadas por estudiante","mediante predicciones externas emparejadas por estudiante"))
change(99,"La validez externa se restringió a las ocho encuestas escolares realizadas entre 2010 y 2019 y a estudiantes de 13 a 15 años. Como cada país aportó una sola encuesta, las diferencias de país y de año resultaron inseparables. La diversidad contextual dependió de ocho unidades geográficas, pese al elevado número de estudiantes. La evaluación de la vigencia de los patrones observados requiere datos escolares más recientes.")
change(68,"La logística del núcleo alcanzó una AP macro de 0,4286 con pesos de encuesta y de 0,4238 al omitirlos de la evaluación nacional. Al agrupar las predicciones con igual peso total por país se obtuvo 0,4228. Estas estimaciones respondieron a formas distintas de resumir el rendimiento. La Figura 3 compara las familias en los mismos estudiantes de cada país. El ensamble promedió 0,4284 de AP y CatBoost, seleccionado para el ajuste final por su rendimiento interno, tampoco superó sistemáticamente a la logística.")
change(76,"Uruguay presentó la mayor ganancia al añadir las preguntas familiares y escolares (0,0663), y Bolivia la menor (0,0156). El incremento conjunto fue compatible con los estudios sobre apoyo parental y conexión escolar (Cohen y Stutts, 2023; DelFerro et al., 2024; Diggs et al., 2025; Niu et al., 2025; Verlenden et al., 2025). En cambio, mantener las variables y ampliar la cohorte de cinco a ocho países cambió la AP en −0,0013 (IC 95 % de −0,0050 a 0,0021) en los mismos países externos. Ampliarla de cinco a siete con el bloque familiar la cambió en −0,0002 (−0,0027 a 0,0031). Ambos intervalos incluyeron cero, lo que subrayó la necesidad de evaluar la generalización en las poblaciones de interés (Richter et al., 2025).")

def insert_before(index,text,style=None):
    added=p[index].insert_paragraph_before(text,style)
    if not style: added.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
    return added
insert_before(84,"Los umbrales y la pendiente se estimaron en etapas distintas. El umbral para clasificar se fijó en el desarrollo. La pendiente se calculó después en la muestra externa para describir el ajuste de las probabilidades. La Figura 5 muestra las frecuencias observadas dentro de los intervalos de probabilidad con al menos veinte estudiantes.")
insert_before(104,"Agradecimientos","Heading 1")
insert_before(104,"Este estudio utilizó datos de la Global School-based Student Health Survey (GSHS), respaldada por la Organización Mundial de la Salud y los Centros para el Control y la Prevención de Enfermedades de los Estados Unidos. Se reconoce la participación de los equipos nacionales, las escuelas y los estudiantes.")
q=insert_before(143,"World Health Organization. (2019). ")
q.paragraph_format.left_indent=Inches(.28);q.paragraph_format.first_line_indent=Inches(-.28)
q.alignment=WD_ALIGN_PARAGRAPH.LEFT
q.add_run("Global School-Based Student Health Survey 2015: Guatemala").italic=True
q.add_run(" [Ficha de microdatos, GTM_2015_GSHS_v01]. WHO NCD Microdata Repository. https://extranet.who.int/ncdsmicrodata/index.php/catalog/488")
for r in q.runs: r.font.size=Pt(10)

figures=[
("Ganancia de precisión promedio del núcleo común frente a edad y sexo","Puntos e intervalos de confianza del 95 % por país y para la media macro."),
("Precisión promedio en desarrollo y en el país excluido","Cada línea compara la media interna de siete países con la evaluación externa en el octavo."),
("Diferencias de precisión promedio entre algoritmos por país","Cada celda muestra la diferencia respecto de la logística con C = 0,1."),
("Curvas de precisión y sensibilidad de las ocho encuestas","Cada curva corresponde a la logística del núcleo común evaluada en el país excluido."),
("Calibración externa de la logística en ocho países","La diagonal indica concordancia. Las barras representan intervalos de confianza del 95 % por conglomerados."),
("Diferencia por sexo en la tasa de falsos negativos","Tasa femenina menos masculina e intervalo de confianza del 95 % por país.")]
for n,(i,j) in enumerate([(57,58),(61,62),(69,70),(79,80),(84,85),(93,94)],1):
    title,note=figures[n-1]
    p[i].clear()
    shp=p[i].add_run().add_picture(str(out/f"Figuras_600dpi/Figura_{n:02}.png"),width=Inches(6.4))
    shp._inline.docPr.set("descr",title)
    p[j].clear()
    p[j].add_run(f"Figura {n}. {title}. ").bold=True
    p[j].add_run(note+" Fuente: elaboración propia.")
    p[j].paragraph_format.keep_with_next=False

countries=dict(ARG="Argentina",BOL="Bolivia",GTM="Guatemala",HND="Honduras",PAN="Panamá",PER="Perú",PRY="Paraguay",URY="Uruguay")
def number(x,n=4): return f"{float(x):.{n}f}".replace(".",",").replace("-","−")
def table(t,rows,widths=None):
    while len(t.rows)<len(rows):t._tbl.append(deepcopy(t.rows[-1]._tr))
    assert len(t.rows)==len(rows)
    if widths:
        for col,w in zip(t.columns,widths):col.width=Inches(w)
    for ri,(vals,row) in enumerate(zip(rows,t.rows)):
        for ci,(cell,value) in enumerate(zip(row.cells,vals,strict=True)):
            if widths:cell.width=Inches(widths[ci])
            par=cell.paragraphs[0]
            par.clear()
            r=par.add_run(value);r.bold=ri==0;r.font.size=Pt(9.5);r.font.name="Times New Roman"
def inc(row,a,b):return number(row[a])+" a "+number(row[b])
def csvrows(file):
    with file.open(encoding="utf-8-sig",newline="") as stream: return list(csv.DictReader(stream))
fam={r["country"]:r for r in csvrows(run/"comparaciones_pareadas.csv") if r["comparison"]=="core_vs_familiar_5paises"}
primary={r["country"]:r for r in data["H1"]}
rows=[["País","ΔAP núcleo¹","IC 95 %","ΔAP familiar²","IC 95 %"]]
for c in [*countries,"Macro"]:
    a=primary[c if c in countries else "Macro · ocho países"]
    b=fam.get(c if c in countries else "MACRO_paises_fijos")
    rows.append([countries.get(c,"Media macro"),number(a["delta_AP"]),inc(a,"low","high"),number(b["delta"]) if b else "—",inc(b,"ci_low","ci_high") if b else "—"])
table(doc.tables[3],rows,[1.08,.88,1.62,1.02,1.8])
change(75,"¹ Núcleo frente a edad y sexo en ocho países. ² Bloque completo frente al núcleo en cinco países. El guion indica que el contraste no se estimó. Fuente: elaboración propia.")
rows=[[c.text for c in doc.tables[4].rows[0].cells]]
for r in data["metricas"]:
    if r["model"]=="B1": rows.append([countries[r["country"]]]+[number(r[k],3) for k in ("threshold","sensitivity","specificity","PPV","Brier","slope","calibration_bias")])
table(doc.tables[4],rows)
features=dict(sex_code="Sexo registrado",alcohol_30d="Consumo de alcohol",physical_attack="Agresión física recibida",close_friends="Amistades cercanas",sedentary_time="Sedentarismo",food_insecurity="Inseguridad alimentaria",physical_fight="Peleas",physical_activity="Actividad física",age_years="Edad")
rows=[["Variable","Media |SHAP|","Pérdida media de AP","Pérdida mínima","Pérdida máxima"]]
for k,label in features.items():
    obs=[r for r in data["explicacion"] if r["variable"]==k]
    assert {r["country"] for r in obs}==set(countries) and {r["n"] for r in obs}=={"512"}
    loss=[float(r["permutation_delta_AP"]) for r in obs]
    rows.append([label,*[number(v) for v in (mean(float(r["mean_abs_SHAP"]) for r in obs),mean(loss),min(loss),max(loss))]])
table(doc.tables[5],rows,[1.75,1.0,1.28,1.15,1.22])
for t in doc.tables:
    for row in t.rows:
        for c in row.cells:
            for par in c.paragraphs:par.paragraph_format.keep_with_next=False
    for cell in t.rows[-1].cells:cell.paragraphs[-1].paragraph_format.keep_with_next=True
doc.core_properties.title=p[0].text
from docx.opc.constants import RELATIONSHIP_TYPE as RT
used={node.get(qn("r:embed")) for node in doc.element.xpath(".//a:blip")}
for rid,rel in list(doc.part.rels.items()):
    if rel.reltype==RT.IMAGE and rid not in used: del doc.part.rels[rid]
doc.settings.element.append(OxmlElement("w:doNotAutoCompressPictures"))
assert all(150<=len(p[i].text.split())<=200 for i in (5,8))
assert len(doc.tables)==len(doc.inline_shapes)==6
dest=out/"Articulo_P07_ocho_paises.docx"
doc.save(dest)
with ZipFile(out/"Figuras_P07_600dpi.zip","w",ZIP_DEFLATED) as archive:
    for filepath in sorted((out/"Figuras_600dpi").glob("Figura_*.png")):archive.write(filepath,filepath.name)
print(dest)

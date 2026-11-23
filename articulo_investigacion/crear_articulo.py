from pathlib import Path
import json,csv,re,hashlib,ast,html
from docx import Document
from docx.shared import Cm,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT=Path(__file__).resolve().parent
ROOT=OUT.parent
RUN=ROOT/'entrenamiento_real/runs/20261117T222805Z_88089e7c'
PROTO=RUN/'protocolo'
def read(path):
 with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def fnum(v,d=4):return f'{float(v):.{d}f}'.replace('.',',')
def integer(v):return f'{int(v):,}'.replace(',',' ')
def interval(a,b,d=4):return f'{fnum(a,d)} a {fnum(b,d)}'

doc=Document()
sec=doc.sections[0]
sec.page_width=Cm(21);sec.page_height=Cm(29.7)
sec.top_margin=Cm(2);sec.bottom_margin=Cm(2)
sec.left_margin=Cm(2.2);sec.right_margin=Cm(2.2)
sec.header_distance=Cm(.8);sec.footer_distance=Cm(.9)
for name in ['Normal','Title','Heading 1','Heading 2','Heading 3','Caption']:
 s=doc.styles[name];s.font.name='Times New Roman';s.font.color.rgb=RGBColor(0,0,0)
 s.font.size=Pt(11)
 s.paragraph_format.space_after=Pt(6)
 s.paragraph_format.line_spacing=1.12
doc.styles['Normal'].paragraph_format.widow_control=True
for name,size in [('Title',16),('Heading 1',13),('Heading 2',11.5)]:
 s=doc.styles[name];s.font.size=Pt(size);s.font.bold=True
 s.paragraph_format.space_before=Pt(10);s.paragraph_format.keep_with_next=True
doc.styles['Title'].paragraph_format.space_before=Pt(0)
doc.styles['Caption'].font.size=Pt(9.5);doc.styles['Caption'].font.italic=False;doc.styles['Caption'].font.bold=False
for st in doc.styles:
 for node in st.element.xpath('.//w:rFonts'):
  for attr in list(node.attrib):
   if 'Theme' in attr:del node.attrib[attr]
 for node in st.element.xpath('.//w:pBdr'):node.getparent().remove(node)
foot=sec.footer.paragraphs[0];foot.alignment=WD_ALIGN_PARAGRAPH.CENTER
run=foot.add_run();run.font.size=Pt(9)
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');run._r.addnext(field)
doc.core_properties.title='Detección de ideación suicida adolescente mediante aprendizaje automático y determinantes sociales con evaluación entre países'
doc.core_properties.subject='Artículo original basado en resultados empíricos GSHS de P07'
doc.core_properties.author=''
doc.core_properties.keywords='GSHS; salud mental adolescente; determinantes sociales; aprendizaje automático'
table_log=[]
def ptext(text,style=None):
 p=doc.add_paragraph(text,style)
 if not style:p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
 return p
def table(n,title,headers,rows,widths,note):
 p=ptext(f'Tabla {n}. {title}','Caption');p.runs[0].bold=True;p.paragraph_format.keep_with_next=True
 t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
 for col,w in zip(t.columns,widths):col.width=Cm(w)
 for i,h in enumerate(headers):t.rows[0].cells[i].text=h
 repeat=OxmlElement('w:tblHeader');t.rows[0]._tr.get_or_add_trPr().append(repeat)
 for row in rows:
  cells=t.add_row().cells
  for c,txt in zip(cells,row):c.text=str(txt)
 for ri,row in enumerate(t.rows):
  trPr=row._tr.get_or_add_trPr();trPr.append(OxmlElement('w:cantSplit'))
  for ci,c in enumerate(row.cells):
   c.width=Cm(widths[ci]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
   tc=c._tc.get_or_add_tcPr();b=OxmlElement('w:tcBorders')
   for side in ['top','left','bottom','right']:
    e=OxmlElement('w:'+side);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');b.append(e)
   tc.append(b)
   mar=OxmlElement('w:tcMar')
   for side,val in [('top','75'),('bottom','75'),('left','80'),('right','80')]:
    e=OxmlElement('w:'+side);e.set(qn('w:w'),val);e.set(qn('w:type'),'dxa');mar.append(e)
   tc.append(mar)
   if ri==0:
    sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'E9EDF0');tc.append(sh)
   for p in c.paragraphs:
    p.paragraph_format.space_after=Pt(0);p.paragraph_format.line_spacing=1.02
    p.paragraph_format.keep_with_next=True
    p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==0 or n==2 else WD_ALIGN_PARAGRAPH.CENTER
    for r in p.runs:r.font.name='Times New Roman';r.font.size=Pt(9 if n!=2 else 9.5);r.bold=ri==0
 p=ptext('Nota. '+note,'Caption');p.paragraph_format.space_before=Pt(4)
 table_log.append({'n':n,'title':title,'headers':headers,'rows':rows,'note':note})
def table1():
 audit=read(RUN/'auditoria_cohorte.csv');eda={x['country']:x for x in read(RUN/'eda_por_pais.csv')}
 names={'ARG':'Argentina','BOL':'Bolivia','GTM':'Guatemala','HND':'Honduras','PAN':'Panamá','PRY':'Paraguay','PER':'Perú','URY':'Uruguay'}
 rows=[]
 for x in audit:
  code=x['survey_id'][:3];e=eda[code]
  rows.append([names[code],x['survey_id'][3:],integer(x['n_raw']),integer(x['n_eligible']),integer(x['events']),fnum(float(e['prevalence_unweighted'])*100,1),fnum(float(e['prevalence_weighted'])*100,1)])
 rows.append(['Total','2010–2019','83 206','50 883','10 683','21,0','—'])
 table(1,'Encuestas y composición de la cohorte de 13 a 15 años',['País','Año','Registros iniciales','Elegibles','Y1 positivos','Y1 sin peso %','Y1 con peso %'],rows,[2.5,1.2,2.35,2.2,2.1,2.1,2.15],
 'Y1: ideación suicida autorreportada en doce meses. Conteos sin ponderar; proporciones ponderadas dentro de cada encuesta. No se calculó una prevalencia regional. Los años y marcos nacionales difirieron. Fuente: auditoria_cohorte.csv y eda_por_pais.csv de la ejecución activa.')
def table2():
 rows=[
 ['Edad y sexo','Edad observada; sexo del cuestionario','Edad 13–15; sexo 1 o 2','Demográfico, núcleo, completo'],
 ['Inseguridad alimentaria','Hambre por falta de alimentos en casa; 30 días','Frecuencia de 1 a 5','Núcleo, completo'],
 ['Agresión física y peleas','Frecuencia de cada experiencia; 12 meses','Dos variables de 1 a 8','Núcleo, completo'],
 ['Amistades cercanas','Número de amistades cercanas','Categorías de 1 a 4','Núcleo, completo'],
 ['Alcohol','Días con consumo; 30 días','Categorías de 1 a 7','Núcleo, completo'],
 ['Actividad física','Días con al menos 60 min; 7 días','Categorías de 1 a 8','Núcleo, completo'],
 ['Sedentarismo','Tiempo sentado en un día habitual','Categorías de 1 a 6','Núcleo, completo'],
 ['Absentismo','Ausencia sin permiso; 30 días','Categorías de 1 a 5','Completo'],
 ['Apoyo de pares','Amabilidad y ayuda de estudiantes; 30 días','Frecuencia de 1 a 5','Completo'],
 ['Supervisión parental','Comprobación de tareas; 30 días','Frecuencia de 1 a 5','Completo'],
 ['Comprensión parental','Comprensión de problemas; 30 días','Frecuencia de 1 a 5','Completo'],
 ['Conocimiento parental','Conocimiento del tiempo libre; 30 días','Frecuencia de 1 a 5','Completo']]
 for row in rows:row[3]=row[3].replace('Demográfico, núcleo, completo','D, N, C').replace('Núcleo, completo','N, C').replace('Completo','C')
 table(2,'Operacionalización y disponibilidad de los predictores',['Dominio','Contenido y ventana','Codificación original','Bloque'],rows,[3.3,6.1,3.6,1.6],
 'D: demográfico; N: núcleo; C: completo. Las filas agrupan catorce variables: edad y sexo son dos; agresión y peleas son dos. Los códigos se expandieron como categorías, sin asumir distancias ordinales. Ausencia individual: categoría explícita. El bloque completo añadió cinco preguntas no disponibles en Guatemala. País, año y otros desenlaces no se utilizaron como predictores. Fuente: armonizacion.csv, variables.json y real_data_training.py.')
def table3():
 names={'B0_logreg':'Logística demográfica','B1_logreg_C01':'Logística C=0,1','B1_logreg_C1':'Logística C=1','B1_random_forest':'Random Forest','B1_xgboost':'XGBoost','B1_catboost':'CatBoost','B1_mlp':'MLP','seleccion_anidada':'Selección anidada'}
 data={r['model']:r for r in read(RUN/'gshs8_core/resumen_macro.csv')}
 rows=[[v]+[fnum(data[k][m]) for m in ['ap','roc_auc','brier','sensitivity','specificity']] for k,v in names.items()]
 table(3,'Rendimiento macro externo en ocho países con el núcleo común',['Modelo o estrategia','AP','ROC-AUC','Brier','Sens.','Espec.'],rows,[5.1,2,2.3,2,1.65,1.65],
 'AP: average precision; ROC-AUC: área bajo la curva ROC; MLP: perceptrón multicapa; Sens.: sensibilidad; Espec.: especificidad. Cada país contribuyó por igual al promedio y las métricas nacionales utilizaron pesos de encuesta. Sensibilidad y especificidad correspondieron a umbrales elegidos internamente por partición. La selección anidada pudo elegir distintas familias; no representa un único ajuste. No se dispuso de IC de estas medias en el artefacto. Fuente: gshs8_core/resumen_macro.csv.')
def table4():
 data=read(RUN/'comparaciones_pareadas.csv');rows=[]
 names={'ARG':'Argentina','BOL':'Bolivia','PER':'Perú','PRY':'Paraguay','URY':'Uruguay'}
 for r in data:
  if r['comparison']=='core_vs_familiar_5paises' and r['country'] in names:
   rows.append([names[r['country']],fnum(r['ap_a']),fnum(r['ap_b']),fnum(r['delta']),interval(r['ci_low'],r['ci_high'])])
 for key,label,aa,bb in [('core_vs_familiar_5paises','Macro añadir familia',.4347,.4702),('5_vs_8_mismas_variables_logistica','Macro ampliar 5→8',None,None),('5_vs_7_familiar','Macro ampliar 5→7',None,None)]:
  r=next(x for x in data if x['comparison']==key and x['country']=='MACRO_paises_fijos')
  rows.append([label,fnum(aa) if aa else '—',fnum(bb) if bb else '—',fnum(r['delta']),interval(r['ci_low'],r['ci_high'])])
 table(4,'Incrementos pareados de AP en los mismos cinco países externos',['Comparación o país','AP núcleo','AP familiar','Diferencia','IC95 % de diferencia'],rows,[4.6,2.1,2.1,2.1,3.7],
 'Las primeras seis filas comparan núcleo común con bloque completo y la misma selección logística. Las dos últimas cambian países de desarrollo conservando variables, estrategia logística y estudiantes externos; no corresponden a las columnas núcleo/familiar. Diferencias B−A, 44 837 estudiantes externos en conjunto. IC percentiles de 500 remuestreos de PSU dentro de estrato, con modelos y países fijos; estratos de una PSU fijos. Fuente: comparaciones_pareadas.csv.')
def table5():
 names={'ARG':'Argentina','BOL':'Bolivia','PER':'Perú','PRY':'Paraguay','URY':'Uruguay'}
 rows=[]
 for r in read(PROTO/'metricas.csv'):
  if r['model']=='B1':rows.append([names[r['country']]]+[fnum(r[m],3) for m in ['threshold','sensitivity','specificity','PPV','Brier','slope','calibration_bias']])
 table(5,'Funcionamiento externo de la logística familiar en cinco países',['País','Umbral','Sens.','Espec.','VPP','Brier','Pendiente','Sesgo'],rows,[3.1,1.6,1.6,1.6,1.6,1.6,1.8,1.8],
 'Sens.: sensibilidad; Espec.: especificidad; VPP: valor predictivo positivo. Sesgo: media ponderada de probabilidad menos proporción observada, ideal 0. Pendiente de recalibración diagnóstica, ideal 1; no se aplicó al modelo externo. Umbrales ajustados internamente; resultados del bloque completo de cinco países, no de ocho países. Fuente: protocolo/metricas.csv.')
def table6():
 exp=read(PROTO/'explicacion.csv');ranks=[r for r in read(PROTO/'estabilidad.csv') if r['model']=='B1_catboost']
 import statistics
 names={'sex_code':'Sexo registrado','alcohol_30d':'Alcohol','physical_attack':'Agresión física recibida','close_friends':'Amistades cercanas','sedentary_time':'Sedentarismo','food_insecurity':'Inseguridad alimentaria','physical_fight':'Peleas','physical_activity':'Actividad física','age_years':'Edad'}
 rows=[]
 for k,v in names.items():
  x=[r for r in exp if r['variable']==k];rr=[float(r['rank']) for r in ranks if r['variable']==k]
  rows.append([v,fnum(statistics.mean(float(r['mean_abs_SHAP']) for r in x)),fnum(statistics.mean(float(r['permutation_delta_AP']) for r in x)),fnum(statistics.median(rr),1),f'{min(rr):g}–{max(rr):g}'])
 table(6,'Magnitud de atribuciones y estabilidad de importancia de CatBoost',['Variable','Media |SHAP|','Pérdida media AP','Rango mediano','Rango mín.–máx.'],rows,[5,2.7,3.1,2.3,2.5],
 'Media simple de cinco países, con ponderación de encuesta dentro de país. SHAP y pérdida de AP: muestras externas de 512 estudiantes por país; rangos: muestras separadas de 384 estudiantes y una permutación por variable. SHAP en log-odds, con indicadores agrupados por variable; no indica dirección ni causalidad. Rango 1 indica mayor pérdida de AP. Una pérdida negativa puede surgir por variación de la permutación. Fuente: explicacion.csv y estabilidad.csv.')

captions={
1:('Aporte del bloque completo frente al modelo demográfico','Diferencias pareadas de AP para regresión logística con catorce predictores frente a edad y sexo. Evaluación por exclusión de país en 44 837 estudiantes de cinco países. Barras: IC95 % percentiles de 500 remuestreos de PSU dentro de estrato, con predicciones fijas. La media asignó igual peso a cada país. Fuente: módulo 7, Aporte de los predictores, protocolo/H1.csv.'),
2:('Comprensión parental e ideación autorreportada en Argentina','Encuesta 2018, adolescentes de 13–15 años con Y1 observado y respuesta parental disponible. Comprensión en 30 días; ideación en doce meses. n por categoría: 7 487; 6 075; 5 807; 5 823; 8 779. Se omitieron 1 450 sin respuesta parental. Barras: IC95 % aproximados por linealización de proporciones ponderadas. Asociación sin ajuste, sin interpretación causal. Fuente: módulo 7, Familia, escuela y alimentación, cohorte de la ejecución activa.'),
3:('Heterogeneidad de diferencias entre familias de modelos','Cada celda representa AP del algoritmo menos AP de logística C=0,1 en el mismo país excluido, con nueve predictores comunes. Colores centrados en cero; valores puntuales sin IC. El ensamble fue la media de probabilidades de cinco familias y tuvo carácter exploratorio. MLP: perceptrón multicapa. Fuente: módulo 7, Comparación de algoritmos, protocolo/H4.csv.'),
4:('Curvas externas de precisión y sensibilidad del bloque familiar','Cinco países iniciales; cada curva procedió de un modelo logístico entrenado sin ese país. Precisión: proporción de positivos entre señalados; sensibilidad: fracción recuperada de positivos. Referencias de prevalencia ponderada: Argentina 21,6 %, Bolivia 22,1 %, Perú 19,5 %, Paraguay 14,0 % y Uruguay 20,1 %. Curvas de hasta 201 puntos; AP se calculó con las predicciones completas. Fuente: módulo 7, curvas externas con todos los países, protocolo/curvas.csv.'),
5:('Calibración externa del bloque familiar con incertidumbre por conglomerados','Probabilidad media frente a proporción ponderada observada en intervalos de amplitud 0,1, con al menos veinte estudiantes. Diagonal: concordancia ideal. Barras: IC95 % percentiles de 500 remuestreos de PSU dentro de estrato, modelos fijos. No son intervalos de riesgo individual. Las cinco encuestas y los modelos correspondieron a la Figura 4. Fuente: módulo 7, calibración externa con todos los países, protocolo/calibracion.csv.'),
6:('Diferencias por sexo en la tasa de falsos negativos','FNR femenina menos masculina con umbral logístico familiar elegido internamente en cada partición de cinco países. Valores negativos indicaron menor proporción de positivos omitidos en el grupo femenino. Barras: IC95 % condicionales por remuestreo de conglomerados. La diferencia coexistió con mayor FPR femenina; no certificó equidad. Sexo según códigos de encuesta, sin inferencia de identidad de género. Fuente: módulo 7, Brechas por sexo y edad, protocolo/equidad.csv.')}
def figure(n):
 p=doc.add_paragraph();p.paragraph_format.space_after=Pt(3);p.paragraph_format.keep_with_next=True
 p.alignment=WD_ALIGN_PARAGRAPH.CENTER
 shape=p.add_run().add_picture(str(OUT/f'figuras/Figura_{n:02}.png'),width=Cm(16.5))
 shape._inline.docPr.set('descr',captions[n][0])
 p=ptext(f'Figura {n}. {captions[n][0]}. {captions[n][1]}','Caption')
 p.paragraph_format.space_after=Pt(10)

refs=json.loads((OUT/'referencias_finales.json').read_text(encoding='utf-8'))
cross={r['doi']:r.get('record',{}) for r in json.loads((OUT/'crossref_verificacion.json').read_text(encoding='utf-8'))}
letters={}
def authors(r):
 arr=[]
 for a in r['authorList']['author']:
  if a.get('collectiveName'):arr.append(a['collectiveName']);continue
  last=a.get('lastName',a.get('fullName',''));ini=a.get('initials','')
  arr.append(last+', '+ ' '.join(v+'.' for v in ini))
 if len(arr)>20:return ', '.join(arr[:19])+', … '+arr[-1]
 if len(arr)==1:return arr[0]
 return ', '.join(arr[:-1])+', & '+arr[-1]
def bibparts(r):
 ji=r['journalInfo'];cr=cross.get(r['doi'],{})
 journal=html.unescape(cr.get('container-title',[ji['journal']['title']])[0])
 year=r['pubYear']+letters.get(r['id'],'')
 title=html.unescape(r['title']).rstrip('.')
 sentence_titles={
 '37222862':'Interpersonal well-being and suicidal outcomes in a nationally representative study of adolescents: A translational study',
 '38949835':'The role of family support in moderating mental health outcomes for LGBTQ+ youth in primary care',
 '37982927':'Unveiling adolescent suicidality: Holistic analysis of protective and risk factors using multiple machine learning algorithms',
 '38758578':'Machine learning-based prediction of suicidal thinking in adolescents by derivation and validation in 3 independent worldwide cohorts: Algorithm development and validation study',
 '39159893':'Neighborhood-level social determinants of health and adolescent mental health',
 '38085207':'Food insecurity, adolescent suicidal thoughts and behaviors, and country-level context: A multi-country cross-sectional analysis',
 '40542804':'Examination of school connectedness as a protective factor for adolescent mental health amidst the COVID-19 pandemic'}
 title=sentence_titles.get(r['id'],title)
 if title=='From Local Explanations to Global Understanding with Explainable AI for Trees':title='From local explanations to global understanding with explainable AI for trees'
 if r['id']=='37822798':journal='Frontiers in Psychiatry'
 vol=ji.get('volume','');iss=ji.get('issue','');pages=r.get('pageInfo') or cr.get('article-number','')
 match=re.fullmatch(r'(\d+)-(\d+)',pages)
 if match:
  a,b=match.groups()
  if len(b)<len(a):b=a[:len(a)-len(b)]+b
  pages=a+'–'+b
 return authors(r)+f' ({year}). '+title+'. ',journal+(', '+vol if vol else ''),(f'({iss})' if iss else '')+(f', {pages}' if pages else '')+'. ', 'https://doi.org/'+r['doi']
def references():
 for r in sorted(refs,key=lambda r:(tuple((a.get('lastName',a.get('collectiveName','')).lower(),a.get('initials','')) for a in r['authorList']['author']),r['pubYear'])):
  if r['authorList']['author'][0].get('lastName')=='Zang':who_reference()
  p=doc.add_paragraph();p.paragraph_format.left_indent=Cm(.7);p.paragraph_format.first_line_indent=Cm(-.7)
  p.paragraph_format.space_after=Pt(6);p.paragraph_format.line_spacing=1.05
  a,b,c,d=bibparts(r);p.add_run(a);p.add_run(b).italic=True;p.add_run(c);p.add_run(d)
  for run in p.runs:run.font.size=Pt(10)
def who_reference():
 p=ptext('World Health Organization. (s. f.). Global school-based student health survey. https://www.who.int/teams/noncommunicable-diseases/surveillance/systems-tools/global-school-based-student-health-survey')
 p.alignment=WD_ALIGN_PARAGRAPH.LEFT
 p.paragraph_format.left_indent=Cm(.7);p.paragraph_format.first_line_indent=Cm(-.7)
 for r in p.runs:r.font.size=Pt(10)

source=(OUT/'manuscrito.md').read_text(encoding='utf-8')
for old,new in [('Collins et al., 2024a','Collins, Dhiman, et al., 2024'),('Collins et al., 2024b','Collins, Moons, et al., 2024'),('Riley et al., 2024a','Riley, Archer, et al., 2024'),('Riley et al., 2024b','Riley, Snell, et al., 2024'),('Riley et al. (2024a)','Riley, Archer, et al. (2024)'),('Riley et al. (2024b)','Riley, Snell, et al. (2024)')]:source=source.replace(old,new)
source=source.replace('0,0286–0,0413; Tabla 4','0,0286–0,0413')
tables={1:table1,2:table2,3:table3,4:table4,5:table5,6:table6}
for block in source.split('\n\n'):
 block=block.strip()
 if not block:continue
 if block.startswith('# '):ptext(block[2:],'Title')
 elif block.startswith('### '):ptext(block[4:],'Heading 2')
 elif block.startswith('## '):
  p=ptext(block[3:],'Heading 1')
  if block.startswith('## Adolescent'):
   p.style=doc.styles['Normal'];p.runs[0].italic=True;p.runs[0].font.size=Pt(13)
 elif re.fullmatch(r'\{TABLE\d\}',block):tables[int(block[-2])] ()
 elif re.fullmatch(r'\{FIGURE\d\}',block):figure(int(block[-2]))
 elif block=='{REFERENCES}':references()
 else:ptext(block.replace('\n',' '))

doc.save(OUT/'Articulo_P07_salud_mental_adolescente.docx')
(OUT/'tablas_del_articulo.json').write_text(json.dumps(table_log,ensure_ascii=False,indent=2),encoding='utf-8')
for t in table_log:
 with (OUT/f'Tabla_{t["n"]:02}.csv').open('w',encoding='utf-8-sig',newline='') as f:
  w=csv.writer(f);w.writerow(t['headers']);w.writerows(t['rows'])
modules=[]
for path in ROOT.rglob('*.py'):
 rel=path.relative_to(ROOT)
 if any(part.startswith('.') for part in rel.parts) or rel.parts[0]=='articulo_investigacion':continue
 try:
  content=path.read_bytes();tree=ast.parse(content.decode('utf-8-sig'))
  modules.append({'archivo':str(rel),'sha256':hashlib.sha256(content).hexdigest(),'lineas':len(content.splitlines()),'funciones':[n.name for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))],'imports':[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]})
 except Exception as e:modules.append({'archivo':str(rel),'error':str(e)})
(OUT/'inventario_modulos.json').write_text(json.dumps(modules,ensure_ascii=False,indent=2),encoding='utf-8')
abstracts={}
for label,end in [('Resumen','Palabras clave:'),('Abstract','Keywords:')]:
 text=source.split('## '+label+'\n\n')[1].split('\n\n'+end)[0]
 abstracts[label]=len(text.split())
 print(label,abstracts[label]);assert 150<=abstracts[label]<=200
assert len(doc.tables)==6 and len(doc.inline_shapes)==6
print('Tablas',len(doc.tables),'Figuras',len(doc.inline_shapes),'Referencias',len(refs)+1,'Módulos',len(modules))
print('Palabras texto',len(source.split()))
print('Título palabras',len(source.splitlines()[0][2:].split()))
(OUT/'validacion_estructura.json').write_text(json.dumps({'resumenes_palabras':abstracts,'tablas':6,'figuras':6,'referencias':len(refs)+1,'articulos_2022_2026':sum(int(r['pubYear'])>=2022 for r in refs),'modulos_escaneados':len(modules)},indent=2),encoding='utf-8')

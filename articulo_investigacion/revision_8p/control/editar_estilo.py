from copy import deepcopy
import json
from pathlib import Path
from zipfile import ZipFile
from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

control=Path(__file__).resolve().parent
root=control.parent
src=root/"Articulo_P07_ocho_paises.docx"
dst=root/"Articulo_P07_revision_de_estilo.docx"
doc=Document(src)
pars=doc.paragraphs
metrics=json.loads((control/"tramos_origen.json").read_text(encoding="utf-8-sig"))
tramos={v["text"].strip()[:35]:v for v in metrics}
revised=set()
def set_text(p,text,language):
    first=next((r for r in p.runs if r.text.strip()),None)
    prop=deepcopy(first._r.rPr) if first is not None and first._r.rPr is not None else None
    p.clear()
    r=p.add_run(text)
    if prop is not None:r._r.insert(0,prop)
    rp=r._r.get_or_add_rPr()
    lng=rp.find(qn("w:lang"))
    if lng is None:
        lng=OxmlElement("w:lang")
        rp.append(lng)
    lng.set(qn("w:val"),language)

polish={
    "siete indicadores: pasar hambre":"siete indicadores. Preguntamos por pasar hambre",
    "dos predicciones para un mismo estudiante: la de la logística con edad y sexo (C = 1) y la de la logística con el núcleo común (C = 0,1)":"las predicciones que recibió cada estudiante con la logística de edad y sexo (C = 1) y con la logística del núcleo común (C = 0,1)",
    "una situación distinta: las cinco":"una situación distinta porque las cinco",
    "a la luz de lo que se probó: una sola arquitectura, hasta cien épocas y sin confirmación de convergencia en todos los ajustes":"teniendo en cuenta que probamos una sola arquitectura, con hasta cien épocas, y que no se confirmó la convergencia en todos los ajustes",
    "Para leer la Figura 4 conviene partir de una pregunta: de todos los estudiantes que quedaron por encima de un umbral, ¿cuántos habían respondido «sí»?":"De los estudiantes que quedaron por encima de cada umbral, ¿qué proporción había respondido «sí»?",
    "Las curvas muestran cómo cambió esa proporción":"Las curvas de la Figura 4 muestran cómo cambió esa proporción",
    "En Paraguay ocurrió lo contrario: el modelo":"En Paraguay, en cambio, el modelo",
    "La pregunta aquí fue más acotada: ¿cuánto dependió el modelo de ese indicador cuando ya disponía de otros ocho?":"Aquí examinamos cuánto dependía el modelo de ese indicador cuando ya disponía de otros ocho.",
    "Dos resultados limitan el uso individual de los puntajes: la calibración varió de un país a otro y la proporción de respuestas afirmativas omitidas y de falsas alarmas cambió según el sexo.":"La calibración varió de un país a otro. También cambiaron, según el sexo, la proporción de respuestas afirmativas omitidas y la de falsas alarmas. Ambas diferencias limitan el uso individual de los puntajes.",
    "en los ocho.":"en los ocho países.",
    "pasar hambre por falta de alimentos, haber sufrido una agresión física, participar en peleas, tener amistades cercanas, consumir alcohol, hacer actividad física y pasar tiempo sentado":"el hambre por falta de alimentos, las agresiones físicas recibidas, la participación en peleas, las amistades cercanas, el consumo de alcohol, la actividad física y el tiempo que el estudiante pasaba sentado",
    "Preguntamos por el hambre":"Las preguntas recogieron el hambre",
    "qué relación guarda el vínculo con la escuela con la salud mental a lo largo del tiempo":"qué relación hay, a lo largo del tiempo, entre sentirse parte de la escuela y la salud mental",
    "El modelo identificó una mayor proporción de respuestas afirmativas entre las alumnas, pero también clasificó erróneamente como positivas más respuestas negativas de ese grupo.":"Entre quienes declararon ideación, el modelo identificó a una mayor proporción de alumnas que de alumnos. Entre quienes respondieron «no», la proporción de falsos positivos también fue mayor en las alumnas.",
}
for line in (control/"redaccion.txt").read_text(encoding="utf8").splitlines():
    num,content=line.split("\t",1)
    i=int(num)
    p=pars[i]
    assert i not in revised
    entry=tramos[p.text.strip()[:35]]
    assert len(content.split(" | "))==len(entry["cut"])+1,(i, entry)
    txt=content.replace(" | "," ")
    for a,b in polish.items():txt=txt.replace(a,b)
    set_text(p,txt,"en-US" if i==8 else "es-PE")
    revised.add(i)
assert revised=={i for i,p in enumerate(pars[:115]) if p.style.name=="Normal" and p.text and i not in (1,2,3,6,9)}
assert all(150<=len(pars[i].text.split())<=200 for i in [5,8])
for table in doc.tables:
    for cell in table.rows[0].cells:
        for par in cell.paragraphs:par.paragraph_format.keep_with_next=True
doc.save(dst)
old=Document(src)
new=Document(dst)
assert [p.text for p in new.paragraphs[115:]]==[p.text for p in old.paragraphs[115:]]
assert len(new.tables)==len(new.inline_shapes)==6
cells=lambda d: [[[c.text for c in r.cells] for r in t.rows] for t in d.tables]
assert cells(new)==cells(old)
with ZipFile(src) as z1,ZipFile(dst) as z2:
    for n in z1.namelist():
        if n.startswith("word/media/"):assert z1.read(n)==z2.read(n)
print(len(revised),"párrafos editados.",len(pars[5].text.split()),len(pars[8].text.split()),"palabras en los resúmenes.")

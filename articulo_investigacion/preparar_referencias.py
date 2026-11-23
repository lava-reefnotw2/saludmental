from pathlib import Path
from urllib.request import urlopen
from urllib.parse import urlencode
import json,re,time
OUT=Path(__file__).resolve().parent
IDS='38758578 39085206 37982927 37822798 37595385 36828148 38078537 41292593 40744149 38085207 39159893 38949835 40111557 40497491 39506487 40542804 37222862 38626948 40127903 41107862 40108256'.split()
QS=['EXT_ID:38214617','DOI:10.1002/wps.21160','DOI:10.1136/bmj-2023-074819','DOI:10.1136/bmj-2023-074820','TITLE:"Evaluation of clinical prediction models (part 3)"','TITLE:"Stability of clinical prediction models"','TITLE:"The false hope of current approaches to explainable artificial intelligence in health care"','TITLE:"Leakage and the reproducibility crisis in machine-learning-based science"','DOI:10.1177/0272989X06295361']
known={}
for path in OUT.glob('*.json'):
 try:
  for x in json.loads(path.read_text(encoding='utf-8')).get('resultList',{}).get('result',[]):known[x['id']]=x
 except Exception:pass
refs=[known[i] for i in IDS]
for q in QS:
 u='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urlencode({'query':q,'format':'json','resultType':'core','pageSize':3})
 try:
  rows=json.loads(urlopen(u,timeout=45).read())['resultList']['result']
  for r in rows:
   if r.get('source')=='MED' and r['id'] not in {x['id'] for x in refs}:refs.append(r)
 except Exception as e:print(q,str(e))
(OUT/'referencias_seleccionadas.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2),encoding='utf-8')
for r in refs:
 print('\nID',r['id'],'YEAR',r.get('pubYear'),'DOI',r.get('doi'),'AUTHORS',r.get('authorString'),'TITLE',r.get('title'))
 print(re.sub('<[^>]+>',' ',r.get('abstractText','Sin resumen')))

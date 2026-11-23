from urllib.request import urlopen
from urllib.parse import urlencode
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

OUT=Path(__file__).resolve().parent
QUERIES={
 'adolescencia_ml':'TITLE_ABS:(adolescent OR adolescents) AND TITLE_ABS:("machine learning") AND TITLE_ABS:(suicide OR suicidal) AND FIRST_PDATE:[2022-01-01 TO 2025-12-31]',
 'sociales':'TITLE_ABS:("global school-based" OR "social determinants" OR "parental support" OR "food insecurity") AND TITLE_ABS:(adolescent OR adolescents) AND TITLE_ABS:(suicide OR suicidal OR "mental health") AND FIRST_PDATE:[2022-01-01 TO 2025-12-31]',
 'metodos':'TITLE:("TRIPOD" OR "PROBAST" OR "prediction models" OR "algorithmic fairness") AND FIRST_PDATE:[2022-01-01 TO 2025-12-31]',
 'guia':'TITLE:("TRIPOD+AI" OR "PROBAST+AI" OR "Rethinking explainability" OR "Minimum information about clinical artificial intelligence modeling" OR "DECIDE-AI" OR "clinical prediction models" OR "FUTURE-AI") AND FIRST_PDATE:[2022-01-01 TO 2025-12-31]',
 'contexto':'TITLE:("adolescent mental health" OR "social determinants of mental health" OR "school connectedness" OR "suicide risk prediction") AND FIRST_PDATE:[2022-01-01 TO 2025-12-31]',
}
def get(item):
 name,q=item
 url='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urlencode({'query':q+' AND SRC:MED','format':'json','resultType':'core','pageSize':60})
 raw=urlopen(url,timeout=60).read().decode()
 (OUT/(name+'.json')).write_text(raw,encoding='utf-8')
 return name,json.loads(raw)['resultList']['result']
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=5) as pool:
  for name,rows in pool.map(get,QUERIES.items()):
   print('\n'+name)
   for x in rows[:50]: print(x['id'],x.get('pubYear'),x.get('title'),x.get('doi'))
 (OUT/'busqueda.json').write_text(json.dumps({'fecha':'2026-11-17','fuente':'Europe PMC MEDLINE','consultas':QUERIES,'restriccion':'No se accedió a Scopus ni Web of Science; búsqueda narrativa, no revisión sistemática.'},ensure_ascii=False,indent=2),encoding='utf-8')

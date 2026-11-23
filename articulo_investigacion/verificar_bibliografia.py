from pathlib import Path
from urllib.request import urlopen,Request
from urllib.parse import quote,urlencode
from concurrent.futures import ThreadPoolExecutor
import json
OUT=Path(__file__).resolve().parent
refs=[r for r in json.loads((OUT/'referencias_seleccionadas.json').read_text(encoding='utf-8')) if r['id']!='38214617']
query='DOI:10.1038/s42256-019-0138-9'
url='https://www.ebi.ac.uk/europepmc/webservices/rest/search?'+urlencode({'query':query,'format':'json','resultType':'core'})
extra=json.loads(urlopen(url,timeout=45).read())['resultList']['result']
refs.extend([r for r in extra if r.get('source')=='MED'])
def crossref(r):
 doi=r.get('doi')
 try:
  u='https://api.crossref.org/works/'+quote(doi,safe='')
  d=json.loads(urlopen(Request(u,headers={'User-Agent':'ScientificManuscriptVerification/1.0'}),timeout=45).read())['message']
  return {'doi':doi,'verified':True,'record':d}
 except Exception as e:return {'doi':doi,'verified':False,'error':str(e)}
with ThreadPoolExecutor(max_workers=4) as pool:result=list(pool.map(crossref,refs))
(OUT/'crossref_verificacion.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
(OUT/'referencias_finales.json').write_text(json.dumps(refs,ensure_ascii=False,indent=2),encoding='utf-8')
print('Referencias',len(refs),'Crossref confirmadas',sum(r['verified'] for r in result))
for r in result:
 if not r['verified']: print(r)
print('EXTRA',json.dumps(extra,ensure_ascii=False))

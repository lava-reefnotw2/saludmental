from pathlib import Path
import urllib.request as U,json,hashlib,concurrent.futures
R=Path(__file__).resolve().parent;D=R/'datasets'/'originales'
jobs=[(f'https://ftp.ibge.gov.br/pense/{y}/microdados/{f}.zip',f'PENSE{y}_{f}.zip') for y in [2019,2024] for f in ['Documentacao','Dados']]
for code in ['NY.GDP.PCAP.PP.KD','SI.POV.GINI','SH.XPD.CHEX.GD.ZS','SL.UEM.1524.ZS']:
 jobs.append((f'https://api.worldbank.org/v2/country/PER;ARG;PRY;URY;BOL;BRA;USA/indicator/{code}?date=2009:2025&format=json&per_page=20000',f'WDI_{code}.json'))
def fetch(x):
 u,n=x;p=D/n
 try:
  if not p.exists():
   with U.urlopen(u,timeout=240) as r,p.open('wb') as f:
    while b:=r.read(1024*1024):f.write(b)
  b=p.read_bytes();o=dict(file=n,url=u,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),status='descargado');print(n,len(b),flush=True);return o
 except Exception as e:print(n,str(e),flush=True);return dict(file=n,url=u,status=str(e))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:r=list(ex.map(fetch,jobs))
(R/'descargas_contexto.json').write_text(json.dumps(r,indent=2),encoding='utf-8')

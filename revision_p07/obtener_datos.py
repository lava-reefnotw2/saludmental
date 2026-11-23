from pathlib import Path
import urllib.request, urllib.parse, http.cookiejar, re, json, hashlib, concurrent.futures
ROOT=Path(__file__).resolve().parent
RAW=ROOT/'datasets'/'originales'; RAW.mkdir(parents=True,exist_ok=True)
def get(url,name):
    p=RAW/name
    try:
        if not p.exists():
            with urllib.request.urlopen(url,timeout=100) as r: p.write_bytes(r.read())
        b=p.read_bytes()
        return dict(file=name,url=url,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),status='descargado')
    except Exception as e:return dict(file=name,url=url,status=str(e))
jobs=[]
for y in [2021,2023,2025]:
    for f in [f'XXH{y}_YRBS_Data.dat',f'{y}XXH-SAS-Input-Program.sas',f'{y}XXH-Formats-Program.sas']:
        jobs.append((f'https://www.cdc.gov/yrbs/files/{y}/{f}',f))
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    results=list(ex.map(lambda a:get(*a),jobs))
print(json.dumps(results,indent=2),flush=True)
(ROOT/'descargas_cdc.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
op=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
u='https://extranet.who.int/ncdsmicrodata/index.php/catalog/204/get-microdata'
s=op.open(u,timeout=60).read().decode()
token=re.search(r'name="ncsrf" value="([^"]+)"',s).group(1)
s=op.open(u,urllib.parse.urlencode({'ncsrf':token,'accept':'Accept'}).encode(),timeout=60).read().decode()
(ROOT/'who_accepted.html').write_text(s,encoding='utf-8')
print('WHO LINKS',re.findall(r'href=[\"\']([^\"\']+)[\"\']',s)[-35:],flush=True)
u='https://ftp.ibge.gov.br/PeNSE/2019/'
try:
 s=urllib.request.urlopen(u,timeout=60).read().decode();(ROOT/'ibge.html').write_text(s,encoding='utf-8');print('IBGE',s,flush=True)
except Exception as e:print('IBGE',e)

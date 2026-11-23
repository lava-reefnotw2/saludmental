from pathlib import Path
import urllib.request as U,urllib.parse as P,http.cookiejar,re,json,hashlib,concurrent.futures
R=Path(__file__).resolve().parent; D=R/'datasets'/'originales'
def survey(item):
 sid,tag=item;out=[]
 op=U.build_opener(U.HTTPCookieProcessor(http.cookiejar.CookieJar()))
 base=f'https://extranet.who.int/ncdsmicrodata/index.php/catalog/{sid}'
 try:
  s=op.open(base+'/get-microdata',timeout=90).read().decode()
  m=re.search(r'name="ncsrf" value="([^"]+)"',s)
  if m:s=op.open(base+'/get-microdata',P.urlencode({'ncsrf':m[1],'accept':'Accept'}).encode(),timeout=90).read().decode()
  (R/f'gshs_{tag}_access.html').write_text(s,encoding='utf-8')
  links=[]
  for a in re.findall(r'<a\s[^>]+>',s):
   h=re.search(r'href="([^"]+)"',a);f=re.search(r'data-filename="([^"]+)"',a)
   if h and f and f[1].lower().endswith(('.csv','.dta')):links.append((h[1],f[1]))
  for url,name in dict.fromkeys(links):
   p=D/(tag+'_'+name)
   if not p.exists():p.write_bytes(op.open(url,timeout=120).read())
   b=p.read_bytes();out.append(dict(file=p.name,url=url,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),status='descargado'))
  for ext in ['json','ddi']:
   url=f'https://extranet.who.int/ncdsmicrodata/index.php/metadata/export/{sid}/{ext}';p=D/f'{tag}_metadata.{ext}'
   if not p.exists():p.write_bytes(op.open(url,timeout=90).read())
  print(tag,out,flush=True)
 except Exception as e:out.append(dict(file=tag,status=str(e)));print(tag,str(e),flush=True)
 return out
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:results=sum(ex.map(survey,[(204,'PER2010'),(866,'ARG2018'),(651,'PRY2017'),(945,'URY2019'),(881,'BOL2018')]),[])
(R/'descargas_gshs.json').write_text(json.dumps(results,indent=2),encoding='utf-8')

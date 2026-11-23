from pathlib import Path
import urllib.request as U,hashlib,json
import sys
R=Path(__file__).resolve().parent;sys.path.insert(0,str(R/'vendor'))
import pandas as pd
D=R/'datasets'/'originales';O=R/'datasets'/'tabulares'
p=next(O.glob('*Dicionario*.xls'))
for sh in pd.ExcelFile(p).sheet_names:
 pd.read_excel(p,sheet_name=sh,header=None).to_csv(O/(p.stem+'_'+sh.replace('/','_')+'.csv'),index=False,header=False)
urls={'YRBS2025_guia.pdf':'https://www.cdc.gov/yrbs/files/2025/2025-YRBS-Data-Users-Guide-Final.pdf','YRBS2023_guia.pdf':'https://www.cdc.gov/yrbs/media/pdf/2023/2023_National_YRBS_Data_Users_Guide508.pdf','YRBS2021_guia.pdf':'https://www.cdc.gov/yrbs/files/2021/pdf/2021_YRBS_Data_Users_Guide_508.pdf'}
manifest=[]
for n,u in urls.items():
 try:
  p=D/n;p.write_bytes(U.urlopen(u,timeout=90).read());b=p.read_bytes();manifest.append(dict(file=n,url=u,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),status='descargado'))
  from pypdf import PdfReader
  text='\n'.join(x.extract_text() for x in PdfReader(p).pages);(R/(n+'.txt')).write_text(text,encoding='utf-8');print(n,len(b),flush=True)
 except Exception as e:print(n,e,flush=True)
(R/'descargas_guias.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')

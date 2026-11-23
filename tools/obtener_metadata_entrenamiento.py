"""Download official dictionaries for the three additional local GSHS files."""
from pathlib import Path
import urllib.request, xml.etree.ElementTree as ET, json, hashlib
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / 'entrenamiento_real' / 'metadata'
OUT.mkdir(parents=True, exist_ok=True)
manifest = []
for tag, sid in [('GTM2015',488), ('HND2012',103), ('PAN2018',879)]:
    url = f'https://extranet.who.int/ncdsmicrodata/index.php/metadata/export/{sid}/ddi'
    p = OUT / f'{tag}.ddi'
    if not p.exists():
        p.write_bytes(urllib.request.urlopen(url, timeout=90).read())
    root = ET.parse(p).getroot()
    rows = []
    for v in root.iter():
        if v.tag.split('}')[-1] != 'var': continue
        def txt(tag):
            x = next((a for a in v.iter() if a.tag.split('}')[-1] == tag), None)
            return ' '.join(' '.join(x.itertext()).split()) if x is not None else ''
        cats = []
        for c in v:
            if c.tag.split('}')[-1] == 'catgry':
                cats.append({a.tag.split('}')[-1]: ' '.join(' '.join(a.itertext()).split()) for a in c})
        rows.append(dict(variable=v.get('name').lower(), file=v.get('files'), label=txt('labl'), question=txt('qstnLit'), categories=json.dumps(cats,ensure_ascii=False)))
    pd.DataFrame(rows).to_csv(OUT/f'{tag}_diccionario.csv',index=False)
    manifest.append(dict(tag=tag,url=url,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
print('Metadata descargada y legible')

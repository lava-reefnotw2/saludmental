"""Check local response domains against official DDI of added countries."""
from pathlib import Path
import pandas as pd,json
ROOT=Path(__file__).resolve().parents[1]
rows=[]
for tag,fn in [('GTM2015','Guatemala.csv'),('HND2012','Honduras.csv'),('PAN2018','Panama.csv')]:
    d=pd.read_csv(ROOT/'datasets'/fn);d.columns=d.columns.str.lower()
    dic=pd.read_csv(ROOT/'entrenamiento_real/metadata'/f'{tag}_diccionario.csv').drop_duplicates('variable').set_index('variable')
    for q in ['q1','q2','q6','q15','q16','q24','q27','q35','q49','q52','q53','q54','q55','q56','q57']:
        if q not in d:continue
        cats=json.loads(dic.loc[q,'categories'])
        allowed={float(c['catValu']) for c in cats if c.get('catValu','').isdigit()}
        unexpected=set(d[q].dropna().unique())-allowed
        assert not unexpected,(tag,q,unexpected)
        counts={float(c['catValu']):int(c['catStat']) for c in cats if c.get('catValu','').isdigit() and c.get('catStat','').isdigit()}
        assert all(int((d[q]==k).sum())==v for k,v in counts.items()),(tag,q,'DDI counts differ')
        rows.append(dict(survey_id=tag,variable=q,domain_valid=True,official_count_categories=len(counts),official_counts_match=True,label=dic.loc[q,'label'],question=dic.loc[q,'question']))
pd.DataFrame(rows).to_csv(ROOT/'entrenamiento_real/verificacion_metadata.csv',index=False)
print(len(rows),'variables verificadas frente a dominios y conteos DDI')

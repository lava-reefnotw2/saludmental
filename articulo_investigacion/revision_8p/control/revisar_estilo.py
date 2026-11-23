from pathlib import Path
import re
from collections import Counter
from PIL import Image
import pypdfium2 as pdf
from docx import Document

folder=Path(__file__).resolve().parent
doc=pdf.PdfDocument(folder/"estilo.pdf")
thumbs=[]
for i,page in enumerate(doc,1):
    im=page.render(scale=1.75).to_pil().convert("RGB")
    im.save(folder/f"estilo_p{i:02}.png")
    im.thumbnail((360,505))
    thumbs.append(im)
for first in range(0,len(thumbs),6):
    panel=Image.new("RGB",(1100,1020),"#c6c9d0")
    for j,img in enumerate(thumbs[first:first+6]):
        panel.paste(img,(366*(j%3)+3,510*(j//3)+2))
    panel.save(folder/f"estilo_lote_{first//6}.png")
old=Document(folder.parent/"Articulo_P07_ocho_paises.docx")
new=Document(folder.parent/"Articulo_P07_revision_de_estilo.docx")
for i,(p,q) in enumerate(zip(old.paragraphs,new.paragraphs)):
    if i>=115:break
    nums=lambda s: set(re.findall(r"(?<!\d)\d+(?:[.,]\d+)+(?!\d)",s))
    assert nums(p.text)==nums(q.text),(i,nums(p.text)^nums(q.text))
    cites=lambda s:Counter(re.findall(r"\b(?:19|20)\d\d\b",s))
    if cites(p.text)!=cites(q.text):print("Años",i,cites(p.text),cites(q.text))
print("Cifras decimales verificadas. Páginas",len(doc))

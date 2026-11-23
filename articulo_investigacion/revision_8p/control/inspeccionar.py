from pathlib import Path
from PIL import Image, ImageDraw
import pypdfium2 as pdf
from pypdf import PdfReader
from docx import Document
from zipfile import ZipFile
from hashlib import sha256
base=Path(__file__).resolve().parent
doc=pdf.PdfDocument(base/"revision.pdf")
reader=PdfReader(base/"revision.pdf")
pages=[]
for i,page in enumerate(doc):
    img=page.render(scale=1.5).to_pil().convert("RGB")
    img.save(base/f"pagina_{i+1:02}.png")
    img.thumbnail((355,495))
    pages.append(img)
    txt=reader.pages[i].extract_text() or ""
    print(i+1,txt[:68].replace("\n"," "),"...",txt[-90:].replace("\n"," "))
def sheet(name,imgs,n,w,h):
    panel=Image.new("RGB",(w*n,h*((len(imgs)+n-1)//n)),"#c7c9d0")
    for i,img in enumerate(imgs):
        img.thumbnail((w-8,h-8))
        panel.paste(img,(w*(i%n)+4,h*(i//n)+4))
    panel.save(base/name)
for j in range(0,len(pages),6):sheet(f"lote_{j//6+1}.png",pages[j:j+6],3,365,505)
imgs=[Image.open(f).convert("RGB") for f in sorted((base.parent/"Figuras_600dpi").glob("*.png"))]
sheet("graficos.png",imgs,2,615,380)
file=base.parent/"Articulo_P07_ocho_paises.docx"
with ZipFile(file) as z:
    embedded=[sha256(z.read(n)).hexdigest() for n in z.namelist() if n.startswith("word/media/")]
assert sorted(embedded)==sorted(sha256(f.read_bytes()).hexdigest() for f in (base.parent/"Figuras_600dpi").glob("*.png"))
d=Document(file)
assert len(d.inline_shapes)==6 and all(abs(s.width.inches-6.4)<.001 for s in d.inline_shapes)
assert sum("Fuente: elaboración propia." in p.text for p in d.paragraphs)==12
print("Figuras originales verificadas dentro de Word. Fuentes de 12 elementos verificadas.")

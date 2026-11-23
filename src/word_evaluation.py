"""Build the downloadable Word report from a finished empirical run."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from docx import Document
from docx.shared import Inches, Pt

from discussion_figures import COUNTRIES, VIEWS, comparison
from publication_charts import png600


def report_figures(run, folder):
    from protocol_views import external_figures, explanation_figures
    for i, view in enumerate(VIEWS):
        # Include all four paired comparisons, plus a labeled example for each optional filter.
        for index in range(4 if i == 0 else 1):
            def choose(label, options, **kwargs):
                return options[index if label == 'Comparación pareada' else 0]
            fig, note = comparison(run, view, choose)
            if fig is not None:
                yield view, fig, note
    for fig,note in explanation_figures(folder):
        yield 'Explicación de los modelos', fig, note
    countries = sorted(pd.read_csv(folder/'curvas.csv', usecols=['country']).country.unique())
    for fig, note in external_figures(folder, countries, True):
        yield 'Validación externa por país', fig, note


def write_report(run, folder):
    doc = Document()
    section = doc.sections[0]
    section.left_margin = section.right_margin = Inches(0.85)
    section.header.paragraphs[0].text = 'SP 07   /   Evaluación de modelos con encuestas escolares'
    section.footer.paragraphs[0].text = 'GSHS    ·    Resultados agregados de investigación'
    normal = doc.styles['Normal']
    normal.font.name = 'Calibri'
    normal.font.size = Pt(12)
    doc.add_heading('Comparaciones y evaluación externa', 0)
    doc.add_paragraph('Ideación autorreportada en los últimos doce meses entre estudiantes escolarizados de 13 a 15 años.')
    doc.add_heading('¿Qué aportan los predictores?', 1)
    h1 = pd.read_csv(folder/'H1.csv')
    macro = h1.loc[~h1.country.isin(COUNTRIES)].iloc[0]
    doc.add_paragraph(f'En los cinco países con comparación pareada, el cambio medio de precisión promedio (AP) al añadir el bloque completo a edad y sexo fue {macro.delta_AP:+.3f} (IC95% {macro.low:+.3f} a {macro.high:+.3f}). Cada país tiene el mismo peso en esta media.')
    doc.add_heading('Resultados por encuesta', 2)
    table = doc.add_table(rows=0, cols=3)
    table.style = 'Light Shading Accent 1'
    for row in [('Encuesta', 'Cambio de AP', 'IC95%')] + [
        (COUNTRIES[r.country], f'{r.delta_AP:+.3f}', f'{r.low:+.3f} a {r.high:+.3f}')
        for r in h1.itertuples() if r.country in COUNTRIES]:
        for cell, text in zip(table.add_row().cells, row):
            cell.text = text
    doc.add_paragraph('Cada fila compara predicciones sobre los mismos estudiantes del país excluido. Un intervalo completamente por encima de cero favorece al modelo completo. Los intervalos reflejan el remuestreo de conglomerados dentro de estratos, con modelos fijos.')
    doc.add_heading('Alcance del informe', 2)
    doc.add_paragraph('Las figuras presentan los cuatro contrastes pareados, ejemplos de las cinco preguntas restantes de discusión además de la importancia por permutación, la dependencia parcial y las curvas externas de todos los países disponibles. Las asociaciones de encuesta y las predicciones describen grupos. La evaluación transversal no establece causalidad ni valida decisiones individuales.')
    doc.add_paragraph('Fuente: microdatos nacionales GSHS. Ejecución '+run.name+'.')
    for number, (topic, fig, note) in enumerate(report_figures(run, folder), 1):
        doc.add_page_break()
        doc.add_paragraph(f'FIGURA {number:02d}    /    {topic}', style='Subtitle')
        doc.add_heading(fig.layout.title.text, 1)
        fig.update_layout(title_text='')
        doc.add_picture(BytesIO(png600(fig)), width=Inches(6.4))
        doc.add_heading('Interpretación', 2)
        doc.add_paragraph(note)
    with TemporaryDirectory(dir=folder) as tmp:
        path = Path(tmp)/'evaluacion.docx'
        doc.save(path)
        path.replace(folder/path.name)

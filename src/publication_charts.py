"""Publication-quality exports of the same Plotly figures displayed in Streamlit."""
import io
import re
import unicodedata

import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image


# Streamlit's theme resolves placeholder colors in the browser only.
# Explicit trace colors keep country legends identical in print and on screen.
px.defaults.color_discrete_sequence = [
    "#0072B2", "#D55E00", "#00856A", "#A44990",
    "#6857B7", "#965B18", "#167B87", "#454545",
]


def png600(fig):
    paper = go.Figure(fig)
    paper.update_layout(template="plotly_white", paper_bgcolor="white", plot_bgcolor="white",
                        font=dict(color="#183747"), margin=dict(l=85, r=45, t=85, b=80))
    raster = paper.to_image(format="png", width=960, height=580, scale=4)
    result = io.BytesIO()
    with Image.open(io.BytesIO(raster)) as image:
        image.save(result, format="PNG", dpi=(600, 600))
    return result.getvalue()


def chart(fig, **kwargs):
    title = fig.layout.title.text or "Figura"
    ascii_title = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    filename = re.sub(r"[^a-z0-9]+", "-", ascii_title.lower()).strip("-") + ".png"
    with st.container(horizontal=True, horizontal_alignment="right"):
        st.download_button("PNG · 600 dpi", data=lambda: png600(fig), file_name=filename,
                           mime="image/png", key="png-" + filename, on_click="ignore",
                           help="Descargar esta figura para impresión, con sus filtros y leyenda.")
    display = go.Figure(fig)
    display.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(display, config={"modeBarButtonsToRemove": ["toImage"]}, **kwargs)

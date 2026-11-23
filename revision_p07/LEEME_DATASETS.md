# Revisión P07 y datasets obtenidos

Verificación: 17 de noviembre de 2026. Los originales suministrados no se sobrescribieron. El protocolo revisado está en `PROTOCOLO_SP07_REVISADO_v3.docx`; incluye comentarios de revisión y un anexo con decisiones, conteos y enlaces oficiales.

## Qué usar primero

`datasets/tabulares/GSHS_5paises_13a15_base.csv` contiene **44.837 filas** de cinco encuestas nacionales oficiales, con **9.509 respuestas positivas** a ideación suicida autorreportada. Son conteos sin ponderar, no prevalencias poblacionales. No se entrenaron modelos en esta revisión.

| Encuesta | Filas originales | Elegibles 13–15 con Y1 | Y1 positivo |
|---|---:|---:|---:|
| Perú 2010 | 2.882 | 2.298 | 457 |
| Argentina 2018 | 56.981 | 35.421 | 7.685 |
| Paraguay 2017 | 3.149 | 1.784 | 247 |
| Uruguay 2019 | 3.162 | 1.745 | 341 |
| Bolivia 2018 | 7.931 | 3.589 | 779 |

Perú agrupa las edades de 16 años en adelante: no se inventaron edades exactas. El análisis principal comparable es 13–15; la extensión 13–17 debe excluir Perú. Los archivos subnacionales descargados son complementarios: **no concatenarlos con el nacional**.

La tabla contiene metadatos, diseño, 14 predictores potenciales y cinco etiquetas. Para B0 usar `age_years` y `sex_code`; B1 añade los otros 12 predictores documentados en `GSHS_mapeo_predictores.csv`. Las respuestas conservan códigos categóricos: `physical_activity=1` significa cero días; `sex_code=1` significa varón en estas encuestas. Los números de categoría no son automáticamente magnitudes continuas.

Excluir de X: `survey_id`, `country_iso3`, `survey_year`, `record_key`, `weight`, `psu`, `stratum` y las cinco columnas `y*`. Los campos de diseño sí sirven para ponderación, particiones e incertidumbre. Imputación, codificación y escalado deben ajustarse dentro de los folds. Nunca imputar etiquetas.

## Alternativas realmente obtenidas

| Archivo en tabulares | Filas | Columnas | Papel recomendado |
|---|---:|---:|---|
| YRBS_2021_completo.csv | 17.232 | 237 | Replicación independiente en EE. UU. |
| YRBS_2023_completo.csv | 20.103 | 250 | Replicación y calibración temporal |
| YRBS_2025_completo.csv | 10.710 | 251 | Evaluación temporal posterior |
| PENSE2019_Dados_PENSE2019_MICRODADOS.csv | 165.838 | 306 | Malestar y contexto familiar/escolar |
| PENSE2024_Dados_PENSE2024_MICRODADOS.csv | 153.870 | 430 | Evaluación temporal brasileña |

Los YRBS se convirtieron desde DAT de ancho fijo usando posiciones y decimales de cada programa SAS oficial. Se conservaron los SAS de formatos y las guías PDF. Y1 es Q26 en 2021 y Q27 en 2023/2025; el número de pregunta no es estable entre años. La guía 2025 informa respuesta global de 30,6 %: se debe considerar no respuesta, aunque el archivo sea descargable y legible.

PeNSE **no se concatena como si tuviera el mismo Y1**: B12005 es tristeza en 30 días; B12008 es sentir que la vida no vale la pena en 30 días. No equivale a considerar seriamente intentar suicidarse en 12 meses. Es útil para un subestudio de malestar con contexto escolar. Los tamaños brutos incluyen registros fuera del dominio analítico; aplicar edad, respuesta y diseño conforme a diccionario. Conservar los registros IND_EXPANSAO=0 para el tratamiento del diseño indicado por IBGE. No recodificar indiscriminadamente todo valor ≥4 como positivo: hay códigos 9 en 2019 y -9/-2 en 2024.

WDI_contexto_largo.csv contiene indicadores agregados. WDI_cobertura_GSHS.csv verifica valores del mismo año en los cinco países para PIB per cápita PPA constante, Gini, gasto corriente en salud y desempleo juvenil. Son **cinco contextos**, no 44.837 observaciones independientes del contexto. B2 permanece exploratorio.

## Trazabilidad y reproducción

- `datasets/originales`: descargas del proveedor, sin cambios.
- `datasets/tabulares`: conversiones y base inicial; incluye diccionarios y auditorías.
- `datasets/manifest_descargas.csv`: URL, tamaño, SHA-256 y fecha.
- `datasets/manifest_completo.json`: incluye huellas de derivados.
- `cambios_protocolo.json`: texto anterior, corrección y motivo.
- `validacion_archivos.json`: controles mecánicos ejecutados sobre los archivos.

Los scripts se ejecutan con Python y pandas; para documentos se requiere python-docx, y para el diccionario XLS de 2019, xlrd. En este entorno se utilizó el Python incluido en Codex. Orden: `obtener_datos.py`, `obtener_gshs.py`, `obtener_contexto.py`, `auditar_datos.py`, `documentacion_extra.py`, `preparar_base.py`. La descarga puede exigir aceptar condiciones de OMS. La lectura XLS de documentacion_extra usa el lector local de `vendor`. No ejecutar los scripts de obtención innecesariamente si ya están todos los archivos; verificar hashes primero.

El inventario de alternativas está documentado, pero no se ha armonizado un dataset universal entre GSHS, YRBS y PeNSE. La base GSHS es un punto de partida trazable, no una validación clínica ni una integración ya implementada en Streamlit. Los CSV previos del proyecto son fixtures sintéticos según la documentación/código local y deben quedar separados de los datos empíricos.

## Acceso y publicación

Los datos OMS son de acceso directo con condiciones de uso, incluida restricción de redistribución. **No subir esta carpeta completa a GitHub ni redistribuir los microdatos sin atender esas condiciones**. El anexo del protocolo recoge obligaciones de atribución/publicación; la institución debe resolver aprobación ética y condiciones antes de publicar resultados.

Fuentes: [OMS GSHS](https://extranet.who.int/ncdsmicrodata/index.php/catalog/GSHS/), [CDC YRBS](https://www.cdc.gov/yrbs/data/), [IBGE PeNSE 2019](https://ftp.ibge.gov.br/pense/2019/microdados/), [IBGE PeNSE 2024](https://ftp.ibge.gov.br/pense/2024/microdados/), [API Banco Mundial](https://datahelpdesk.worldbank.org/knowledgebase/articles/898599-indicator-api-queries). Los enlaces específicos de cada descarga están en el manifiesto.

# Detección de ideación suicida adolescente mediante aprendizaje automático y determinantes sociales con evaluación entre países

## Adolescent suicidal ideation detection using machine learning and social determinants with cross country evaluation

Autores y afiliaciones: pendientes de confirmación por el equipo investigador.

ORCID de cada autor y correo del autor de correspondencia: pendientes de incorporación.

## Resumen

Contexto: La identificación de vulnerabilidad adolescente requirió integrar información social sin confundir clasificación concurrente con pronóstico. Objetivo: Se evaluó el aporte de determinantes sociales y conductuales a la detección de ideación suicida autorreportada y su transporte entre países. Métodos: Se analizaron encuestas escolares transversales de ocho países latinoamericanos, con 50 883 estudiantes de 13 a 15 años y 10 683 respuestas positivas. Se compararon cinco familias de aprendizaje automático mediante exclusión de un país, selección interna agrupada, ponderación de encuesta y calibración interna. Se realizaron comparaciones pareadas en cinco países y remuestreo de conglomerados. Resultados: La selección anidada alcanzó precisión promedio de 0,4282 y área bajo la curva ROC de 0,7346. En cinco países, el bloque familiar y escolar superó al demográfico en 0,2136 de precisión promedio; añadirlo al núcleo conductual produjo una mejora de 0,0355, con intervalo de confianza del 95 % de 0,0286 a 0,0413. Persistieron diferencias de calibración y errores por sexo. Conclusiones: La información social aportó señal predictiva transferible dentro de las encuestas estudiadas. Los resultados respaldaron investigación poblacional, pero no acreditaron diagnóstico, predicción futura ni beneficio preventivo individual.

Palabras clave: calibración, encuestas escolares, equidad algorítmica, explicabilidad, vulnerabilidad psicosocial.

## Abstract

Background: Identifying adolescent vulnerability required incorporating social information while distinguishing concurrent classification from prognosis. Objective: We evaluated the contribution of social and behavioral determinants to detecting self-reported suicidal ideation and its transportability across countries. Methods: We analyzed cross-sectional school surveys from eight Latin American countries, including 50,883 students aged 13–15 years and 10,683 positive responses. Five machine learning families were compared using country-held-out evaluation, grouped internal selection, survey weighting, and internal calibration. Paired comparisons in five countries used cluster resampling. Results: Nested selection achieved an average precision of 0.4282 and a receiver operating characteristic area under the curve of 0.7346. In five countries, the family and school feature block outperformed the demographic baseline by 0.2136 average precision units. Adding family and school information to the behavioral core improved average precision by 0.0355, with a 95% confidence interval of 0.0286–0.0413. Calibration differences and sex-related error disparities persisted. Conclusions: Social information provided transferable predictive signal within the studied surveys. These findings supported population research but did not establish individual diagnosis, future prediction, or preventive benefit. Further prospective evaluation and context-specific assessment remained necessary before implementation.

Keywords: algorithmic fairness, calibration, explainability, psychosocial vulnerability, school surveys.

## Introducción

La salud mental adolescente depende de condiciones materiales, relaciones familiares y experiencias escolares que distribuyen de manera desigual la exposición a adversidad y los recursos de apoyo. Los determinantes sociales comprenden tanto procesos estructurales como condiciones próximas de vida; por ello, identificar asociaciones individuales no equivale a explicar el origen de las desigualdades (Kirkbride et al., 2024). En adolescentes, el entorno de oportunidades también se relaciona con diferencias en los diagnósticos registrados, aunque acceso asistencial y detección pueden influir en esas relaciones (Oyegoke et al., 2024).

Las encuestas internacionales permiten examinar esas condiciones fuera de muestras clínicas. El análisis multinacional de Steare et al. (2024) documenta heterogeneidad entre países en la asociación de inseguridad alimentaria con pensamientos y conductas suicidas. Hu et al. (2025) describen asociaciones entre soledad e indicadores suicidas en numerosos contextos. Estos hallazgos sustentan una aproximación social, pero no justifican trasladar una prevalencia, un umbral o una relación estimada de un país a otro sin evaluación explícita.

El apoyo familiar y la conexión escolar representan dimensiones particularmente pertinentes. Cohen y Stutts (2023) identifican información clasificatoria en el cuidado parental y la cohesión familiar; Diggs et al. (2025) y Verlenden et al. (2025) aportan evidencia longitudinal sobre conexión escolar y salud mental. DelFerro et al. (2024) muestran que el apoyo familiar también importa al examinar desigualdades en jóvenes LGBTQ+. Estos resultados no convierten los ítems escolares aislados en escalas clínicas, pero orientan la selección de variables socialmente interpretables.

El aprendizaje automático amplía las posibilidades de integrar estos dominios. Los estudios de Donnelly et al. (2023, 2025), Shin y Kim (2023) y Azasu (2025) exploran determinantes demográficos, sociales y psicológicos en diferentes poblaciones adolescentes. Sin embargo, desenlaces, predictores y esquemas de evaluación difieren. Una clasificación que incorpora síntomas emocionales próximos al desenlace responde a una pregunta distinta de otra restringida a condiciones sociales y conductuales; sus cifras de rendimiento no constituyen una comparación directa de algoritmos.

La evaluación internacional ya cuenta con antecedentes relevantes. Kim et al. (2024) desarrollan y evalúan modelos en tres cohortes nacionales; Haghish et al. (2023, 2024) estudian algoritmos y dominios de riesgo en adolescentes noruegos. Zang et al. (2024) encuentran que modelos complejos no necesariamente superan a la regresión regularizada al transportarse entre fuentes clínicas. Niu et al. (2025) examinan información de adolescentes y familiares. Por tanto, la brecha no consiste en la ausencia de modelos, sino en separar el aporte de la información social del efecto de cambiar algoritmo, población o diseño de validación.

El proyecto P07 permite abordar esa comparación mediante encuestas latinoamericanas armonizadas y evaluación por país. Este estudio tiene como objetivo determinar si los predictores sociales y conductuales mejoran la clasificación concurrente de ideación suicida frente a edad y sexo, cuantificar el incremento atribuible al bloque familiar y escolar, y describir transporte, calibración y errores por subgrupos. La hipótesis principal plantea una mejora de la precisión promedio con información social. Se examinan además la superioridad eventual de modelos no lineales y la estabilidad de las explicaciones, siguiendo criterios de transparencia y evaluación de sesgos de TRIPOD+AI y PROBAST+AI (Collins et al., 2024b; Moons et al., 2025).

## Materiales y métodos

### Diseño y alcance analítico

Se realizó un estudio cuantitativo, observacional y transversal de análisis secundario, con desarrollo y evaluación interna–externa de modelos predictivos. El ámbito geográfico comprendió Argentina, Bolivia, Guatemala, Honduras, Panamá, Paraguay, Perú y Uruguay. Se trabajó con la Global School-based Student Health Survey (GSHS), una encuesta escolar con cuestionarios autoadministrados sobre conductas y factores protectores (World Health Organization, s. f.). Se analizaron los resultados empíricos conservados por la aplicación P07 en la ejecución 20261117T222805Z_88089e7c, completada el 17 de noviembre de 2026. No se sustituyeron resultados por valores supuestos ni se incorporaron los archivos históricos de demostración sintética.

La unidad de observación fue el estudiante; la unidad de separación entre desarrollo y evaluación fue el país. Se distinguieron cuatro experimentos: núcleo común de nueve predictores en ocho países; bloque completo de catorce predictores en siete países; y ambos bloques en los mismos cinco países de comparación —Argentina, Bolivia, Perú, Paraguay y Uruguay—. Este último conjunto permitió estimar incrementos sin cambiar los estudiantes evaluados. La exclusión geográfica aportó evidencia de transporte entre encuestas disponibles, pero no constituyó una evaluación prospectiva independiente de la estrategia completa (Collins et al., 2024a; Riley et al., 2024a).

### Población y construcción de las cohortes

Las fuentes correspondieron a Perú 2010, Honduras 2012, Guatemala 2015, Paraguay 2017, Argentina 2018, Bolivia 2018, Panamá 2018 y Uruguay 2019. Se conservaron los pesos, estratos y unidades primarias de muestreo (PSU) suministrados en los archivos. El muestreo original fue escolar y por conglomerados; la selección de las ocho encuestas para este análisis dependió de disponibilidad y armonización, por lo que no representó un muestreo aleatorio de países latinoamericanos. Se incluyeron estudiantes con edad observada de 13–15 años y respuesta válida al desenlace. Se verificaron dominios de codificación, pesos finitos positivos, identificadores únicos y disponibilidad del diseño muestral.

De 83 206 registros iniciales se excluyeron 30 860 por edad ausente o fuera del intervalo y 1 463 por desenlace ausente después de aplicar el criterio etario. La cohorte final incluyó 50 883 participantes; el bloque familiar de siete países comprendió 47 589 y la comparación controlada de cinco países, 44 837. No se imputaron desenlaces. Se emplearon todos los registros elegibles disponibles, sin cálculo prospectivo de tamaño muestral ni muestreo adicional. La precisión de los análisis por país y subgrupo se consideró limitada por el número efectivo de conglomerados, y no solamente por el total de filas (Riley et al., 2024b).

### Desenlace y operacionalización de predictores

El desenlace primario fue una respuesta afirmativa al ítem nacional armonizado sobre haber considerado seriamente intentar suicidarse durante los doce meses previos. Se codificó como Y1=1 para respuesta afirmativa y Y1=0 para negativa. Se trató de ideación autorreportada retrospectiva, no de diagnóstico psiquiátrico, intento de suicidio observado ni riesgo de un evento futuro. No se estableció validación contra entrevista clínica ni equivalencia psicométrica formal entre países. Los ítems sobre planificación, intentos, soledad y preocupación con alteración del sueño se excluyeron de los predictores principales.

El comparador demográfico utilizó edad y sexo registrado por la encuesta. El núcleo común añadió hambre por insuficiencia de alimentos, agresión física recibida, participación en peleas, amistades cercanas, consumo de alcohol, actividad física y sedentarismo. El bloque familiar y escolar incorporó absentismo, amabilidad o ayuda de pares, supervisión de tareas, comprensión parental y conocimiento parental de las actividades de tiempo libre. La Tabla 2 documentó sus ventanas de recuerdo y tratamiento. Esas medidas representaron condiciones y experiencias próximas; no abarcaron por sí mismas ingreso, desigualdad, etnicidad o políticas nacionales.

Se verificaron etiquetas, categorías y preguntas nacionales contra los diccionarios locales de origen oficial; la coincidencia de números de pregunta no se consideró suficiente. Guatemala careció estructuralmente de los cinco ítems familiares y escolares y se excluyó del análisis de ese bloque. Para los predictores presentes se representó la falta de respuesta como una categoría explícita, codificada mediante un valor centinela y expansión indicadora. El preprocesamiento se ajustó dentro de cada partición; categorías no observadas en desarrollo se ignoraron al codificar la prueba. No se aplicó imputación de medias, sobremuestreo sintético ni balanceo artificial de clases.

### Algoritmos y prevención de contaminación entre particiones

Se compararon regresión logística, Random Forest, XGBoost, CatBoost y perceptrón multicapa (MLP). La logística evaluó C=0,1 y C=1, con un máximo de 600 iteraciones. Random Forest utilizó 120 árboles, profundidad máxima 9 y mínimo de 30 observaciones por hoja. XGBoost utilizó 150 árboles, profundidad 3, aprendizaje 0,05, submuestreo de filas y columnas de 0,85 y penalización lambda de 5. CatBoost utilizó 180 iteraciones, profundidad 4, aprendizaje 0,05 y penalización L2 de 5; recibió las mismas variables indicadoras, sin explotar codificación categórica nativa. MLP utilizó capas de 32 y 16 unidades, alfa=0,1 y máximo de 100 épocas. La semilla general fue 42.

La comparación de cinco familias se ejecutó en el núcleo de ocho países. En los bloques familiares se compararon las dos configuraciones logísticas; no se atribuyó a esos bloques una búsqueda completa de cinco familias. El presupuesto fue acotado y desigual entre familias, sin optimización exhaustiva. Las advertencias de convergencia se suprimieron durante el ajuste en el código conservado; por ello, la comparación no acreditó convergencia de todas las ejecuciones de MLP. El ensamble de media de probabilidades se consideró un análisis exploratorio posterior de las predicciones externas.

Se aplicó exclusión de un país por vez en el nivel externo y GroupKFold de tres grupos de países dentro de cada conjunto de desarrollo. El criterio de selección fue la media de average precision (AP) entre países internos, con ponderación de encuesta dentro de cada país. Los pesos de entrenamiento mantuvieron sus proporciones nacionales y se normalizaron para asignar igual masa a cada país, evitando que Argentina dominara el ajuste por su tamaño. Identificadores, país, año, peso, estrato, PSU y etiquetas de salud mental quedaron fuera de la matriz predictora. Estos controles respondieron al riesgo de contaminación que puede inflar artificialmente el rendimiento de modelos científicos (Kapoor y Narayanan, 2023).

### Calibración y medidas de rendimiento

Para cada candidato se ajustó una regresión logística de calibración sobre el logit de probabilidades internas fuera de muestra. Se seleccionó un umbral entre 0,05 y 0,60, en incrementos de 0,01, minimizando un costo ponderado con relación falso negativo:falso positivo de 3:1. Ese costo fue un supuesto de investigación y no una preferencia clínica validada. Selección, calibración y umbral utilizaron datos internos; el país excluido permaneció ajeno al ajuste. Las mismas predicciones internas contribuyeron a esas tres operaciones, de modo que sus métricas no se trataron como estimaciones independientes del rendimiento final.

La medida principal fue AP, calculada como la suma de precisiones ponderadas por los incrementos de sensibilidad de la curva precisión–sensibilidad. No se equiparó a exactitud ni a integración trapezoidal de esa curva. Se estimaron también área bajo la curva de características operativas del receptor (ROC-AUC), puntuación de Brier, sensibilidad, especificidad y valor predictivo positivo. Las métricas agregadas fueron medias aritméticas entre países; no estimaron un rendimiento continental ponderado por población. Se describieron sesgo de calibración —media de probabilidad menos frecuencia observada—, intercepto y pendiente de recalibración diagnóstica externa. Estos últimos no se utilizaron para corregir las predicciones evaluadas.

Se examinaron curvas de calibración en intervalos de probabilidad de amplitud 0,1, conservando grupos con al menos veinte observaciones. El Brier se interpretó como error probabilístico global y no como prueba suficiente de calibración. La interpretación conjunta de discriminación, calibración y utilidad siguió la orientación metodológica de Riley et al. (2024a). Las curvas mostradas en la web se exportaron con sus mismos datos; la representación de precisión–sensibilidad conservó hasta 201 puntos por curva, mientras que AP se calculó con las predicciones completas.

### Comparaciones pareadas y análisis de patrones

Se estimaron diferencias de AP para el bloque completo frente al demográfico, el bloque familiar frente al núcleo común y la ampliación del número de países. Las comparaciones controladas mantuvieron los mismos estudiantes externos de los cinco países iniciales. Se obtuvieron intervalos percentiles del 95 % mediante 500 remuestreos pareados de PSU dentro de estrato y país, manteniendo las predicciones fijas. Los estratos con una sola PSU permanecieron fijos; no se aplicó corrección por población finita. Los intervalos fueron condicionales a países y modelos ajustados y no incorporaron variabilidad del reentrenamiento. No se realizaron pruebas confirmatorias múltiples ni se interpretó ausencia de diferencia como equivalencia.

Se describieron diferencias de tasas de falsos negativos (FNR), falsos positivos (FPR), AP y calibración por sexo y edad, excluyendo comparaciones con menos de diez positivos o diez negativos por grupo. El sexo se interpretó según los códigos binarios del cuestionario, sin inferir identidad de género. Se resumieron valores absolutos de Shapley additive explanations (SHAP) de CatBoost sobre hasta 512 estudiantes externos por país; las contribuciones de indicadores de una misma variable se sumaron antes de obtener magnitudes absolutas. Se examinó la importancia por permutación y su rango en muestras externas de 384 estudiantes por país. Esas muestras correspondieron a los cinco países iniciales y al núcleo común, no al bloque familiar ni a los ocho países completos.

Las explicaciones se expresaron en log-odds del modelo sin calibrar y se interpretaron como atribuciones predictivas (Lundberg et al., 2020). Se utilizó una permutación por variable y país, por lo que la variabilidad de esa operación no quedó estimada exhaustivamente. Las asociaciones entre comprensión parental e ideación se describieron por categorías con intervalos aproximados por linealización de la proporción dentro de la cohorte elegible. Se omitieron categorías con menos de diez positivos o negativos; los estratos con una PSU aportaron varianza cero. La Figura 2 ilustró Argentina, seleccionada por su mayor tamaño muestral, sin presentarla como patrón universal ni efecto causal.

### Entorno computacional y trazabilidad del despliegue

Se inspeccionaron los módulos del proyecto y se siguió el flujo activo desde app.py hacia real_dashboard.py y los módulos de entrenamiento, análisis y exportación. La aplicación Streamlit se desplegó localmente en 127.0.0.1:8501, con siete módulos de ingesta, calidad, variables, análisis exploratorio, modelos, calibración y comparación. El archivo active.json identificó la ejecución visible. El proceso de entrenamiento conservó cada ejecución y actualizó el puntero activo después de completar verificaciones. Se contrastaron los resultados presentados con CSV, configuración, manifiesto y huellas SHA-256 de las fuentes.

El registro de ejecución consignó Python 3.12.14, NumPy 2.5.5, pandas 3.1.1 y scikit-learn 1.9.1; la procedencia del informe añadió CatBoost 1.2.10 y Streamlit 1.67.0. Las configuraciones completas se conservaron en receta_predefinida.json. Se distinguió el circuito empírico de los módulos antiguos y archivos de demostración. Los análisis Optuna disponibles en subcarpetas posteriores no se integraron en la comparación primaria. La ejecución seleccionada no incluyó YRBS; tampoco incorporó PeNSE, UNICEF ni indicadores macroeconómicos del Banco Mundial en los predictores principales. Las seis figuras procedieron de funciones de visualización de la web; solo se ajustaron rótulos y tipografía para publicación.

### Fuentes bibliográficas y consideraciones éticas

Se realizó una búsqueda narrativa de antecedentes el 17 de noviembre de 2026 mediante Europe PMC/MEDLINE y páginas editoriales, con combinaciones de adolescent, suicidal ideation, machine learning, global school-based, food insecurity, parental support, social determinants, TRIPOD y PROBAST. Se priorizaron artículos de 2022–2025 y se conservaron métodos seminales pertinentes. Se contrastaron DOI y metadatos con Crossref o la editorial. No se ejecutó una búsqueda en las plataformas de suscripción Scopus o Web of Science y no se presentó esta contextualización como revisión sistemática. El registro de consultas y metadatos acompañó los materiales de preparación.

Se trabajó con archivos secundarios sin identificación directa en el análisis y se presentaron únicamente agregados. No se contactó a participantes ni se generaron decisiones individuales. No se encontró entre los insumos una resolución de aprobación o exención ética específica para este análisis; su estado quedó pendiente de acreditación por la institución responsable antes de una eventual presentación editorial. Tampoco se presumieron permisos de redistribución de microdatos. La aplicación se interpretó como entorno de investigación; no se evaluaron intervención, seguridad clínica ni consecuencias de asignar recursos con sus predicciones.

## Resultados y discusión

### Composición y comparabilidad de la muestra

La cohorte incluyó 10 683 respuestas afirmativas, equivalentes al 21,0 % de los registros elegibles sin ponderar. Argentina aportó 69,6 % de las observaciones. Las proporciones ponderadas nacionales variaron de 14,0 % en Paraguay a 22,1 % en Bolivia (Tabla 1). La media simple de las ocho proporciones ponderadas fue 19,0 %; esta cifra describió las encuestas incluidas y no una prevalencia latinoamericana. La combinación de países y años distintos impidió atribuir diferencias observadas exclusivamente a contexto nacional.

{TABLE1}

Las ausencias entre predictores presentes fueron generalmente reducidas, pero alcanzaron 7,28 % para supervisión parental en Uruguay. La ausencia de los cinco ítems familiares en Guatemala fue estructural, no un patrón de no respuesta individual. La separación de bloques preservó esa diferencia (Tabla 2). El análisis de Anika et al. (2025) sobre Panamá respaldó la pertinencia de examinar hambre, vínculos y alcohol; sin embargo, no constituyó replicación independiente respecto de este estudio, al utilizar la misma encuesta nacional de 2018. Asimismo, otros trabajos basados en GSHS pudieron compartir observaciones y no se consideraron confirmaciones externas automáticas.

{TABLE2}

La restricción a estudiantes escolarizados limitó la cobertura de adolescentes fuera de la escuela, potencialmente expuestos a condiciones sociales diferentes. Las ventanas de recuerdo variaron entre predictores y desenlace. Esas diferencias, junto con el autorreporte común, dejaron abierta la posibilidad de sesgo de recuerdo, causalidad inversa y confusión residual. El marco social de Kirkbride et al. (2024) fue útil para interpretar las variables como indicadores de condiciones de vida, evitando reducir las desigualdades a atributos individuales aislados.

### Aporte incremental de la información social

En los cinco países de comparación, el bloque completo alcanzó AP macro de 0,4702 frente a 0,2567 del modelo demográfico. La diferencia fue 0,2136 (IC95 %: 0,2032–0,2246), con incrementos positivos en cada país (Figura 1). Este contraste respaldó la hipótesis principal en las encuestas evaluadas. No demostró que los predictores fueran causas modificables ni que una clasificación más informativa redujera conductas suicidas.

{FIGURE1}

El incremento atribuible a las cinco preguntas familiares y escolares, por encima del núcleo común, fue menor pero consistente: AP macro de 0,4347 a 0,4702, con diferencia de 0,0355 (IC95 %: 0,0286–0,0413; Tabla 4). Esa comparación resultó más específica que contraponer únicamente el bloque completo con edad y sexo, porque mantuvo el algoritmo, los países externos y los estudiantes. La información familiar complementó las señales conductuales sin requerir indicadores de síntomas mentales como predictores.

En Argentina, la proporción ponderada de ideación fue 36,7 % (IC95 % aproximado: 34,6–38,9 %) entre quienes nunca percibieron comprensión parental y 10,8 % (9,8–11,7 %) entre quienes siempre la percibieron (Figura 2). El patrón fue compatible con el papel de las relaciones familiares descrito por Cohen y Stutts (2023), DelFerro et al. (2024) y Niu et al. (2025). Al comparar estudiantes distintos y no ajustar confusores en esta figura, no se estimó el efecto de aumentar la comprensión parental ni se probó mediación.

{FIGURE2}

La amabilidad de pares, el absentismo y las preguntas parentales no reprodujeron escalas completas de conexión escolar o funcionamiento familiar. Por ello, los resultados de Diggs et al. (2025) y Verlenden et al. (2025) aportaron plausibilidad contextual, pero no validaron automáticamente esos ítems. La contribución observada correspondió al bloque conjunto; no permitió asignar la mejora de AP a una sola pregunta ni distinguir una intervención familiar de una escolar.

### Rendimiento de las familias de modelos

En ocho países, la selección anidada alcanzó AP=0,4282, ROC-AUC=0,7346, Brier=0,1344, sensibilidad=0,5147 y especificidad=0,8183. La logística C=0,1, XGBoost y CatBoost obtuvieron AP macro entre 0,4281 y 0,4286 (Tabla 3). Random Forest mostró una AP algo menor y MLP presentó el menor desempeño entre los modelos ampliados. Los resultados no justificaron proclamar superioridad de boosting sobre la logística bajo este presupuesto computacional.

{TABLE3}

Las diferencias por país mostraron que pequeñas ventajas locales no se sostuvieron uniformemente (Figura 3). El ensamble exploratorio obtuvo AP macro de 0,4284 y no estableció una ganancia respecto de la logística. CatBoost fue seleccionado para el ajuste final mediante el criterio interno, no por ocupar el primer lugar en la tabla externa. La selección por partición pudo elegir familias diferentes; su fila resumió una estrategia de selección y no un único modelo fijo evaluado en todos los países.

{FIGURE3}

La cercanía entre algoritmos coincidió con la advertencia empírica de Zang et al. (2024) sobre complejidad y transporte. Los resultados de Kim et al. (2024) y Haghish et al. (2024) no fueron comparadores numéricos directos: utilizaron poblaciones, desenlaces y conjuntos de variables diferentes, incluidos indicadores emocionales o autolesivos relevantes. Haghish et al. (2023) mostraron la factibilidad de excluir ítems suicidas, pero con un conjunto de preguntas y otro desenlace que tampoco equivalieron al núcleo de este estudio. La menor AP de MLP no se interpretó como inferioridad general de redes neuronales, dada la arquitectura única, las cien épocas y la convergencia no acreditada.

### Información adicional y ampliación geográfica

La comparación pareada favoreció añadir el bloque familiar y escolar, mientras que ampliar la cantidad de países de entrenamiento no produjo una mejora media clara en los cinco países externos comunes (Tabla 4). El cambio de cinco a ocho países con núcleo común fue −0,0013 (IC95 %: −0,0050 a 0,0021); el cambio de cinco a siete países con bloque familiar fue −0,0002 (−0,0027 a 0,0031). Estos intervalos abarcaron cero, sin demostrar equivalencia ni descartar beneficios específicos en poblaciones distintas.

{TABLE4}

El bloque familiar de siete países alcanzó AP macro de 0,4661 y ROC-AUC de 0,7552. Su comparación directa con AP=0,4282 del núcleo de ocho países habría mezclado diferencias de variables, países y familias candidatas; por eso, la inferencia sobre incremento se apoyó en el contraste pareado de cinco países. La ampliación de cobertura aportó diversidad geográfica, pero no garantizó mejora predictiva. Este resultado fue compatible con la necesidad de evaluar generalización de manera explícita, en lugar de inferirla del tamaño de la base (Richter et al., 2025).

### Discriminación y calibración fuera del país de desarrollo

Las curvas precisión–sensibilidad del modelo familiar mostraron heterogeneidad entre los cinco países (Figura 4). La precisión disminuyó al recuperar una fracción mayor de positivos, y su interpretación dependió de la frecuencia del desenlace. Una AP mayor no significó que igual proporción de todas las predicciones fuera correcta. Las diferencias de prevalencia y composición limitaron la comparación de AP entre países, aun cuando se utilizaron las mismas variables.

{FIGURE4}

La calibración mostró discrepancias que la discriminación agregada no hizo visibles (Tabla 5 y Figura 5). Bolivia presentó pendiente de 0,753 y sesgo promedio de −0,0249; Paraguay mostró pendiente de 1,156 y sesgo de 0,0208. Las sensibilidades oscilaron entre 0,462 y 0,687, mientras los valores predictivos positivos variaron de 0,359 a 0,461. En consecuencia, una fracción considerable de respuestas positivas no fue recuperada por los umbrales investigados, y muchos estudiantes señalados no habían declarado ideación.

{TABLE5}

{FIGURE5}

Los resultados reforzaron la necesidad de examinar calibración y errores junto con ROC-AUC, según Riley et al. (2024a). Los intervalos por grupos de probabilidad fueron descriptivos y se ampliaron donde hubo menos observaciones. No se estimó el intervalo de cada probabilidad individual. El diagnóstico de recalibración realizado en el país excluido describió desajustes; no se utilizó para mejorar retrospectivamente el desempeño comunicado. Antes de cualquier uso en otra población se requerirían evaluación temporal, recalibración independiente cuando procediera y una valoración de la disponibilidad de apoyo.

### Patrones explicativos y diferencias por sexo

Las atribuciones de CatBoost en los cinco países situaron sexo, alcohol y agresión física entre las variables de mayor magnitud media de SHAP (Tabla 6). El sexo presentó el primer rango mediano por permutación, pero su posición osciló entre primero y quinto; la agresión física varió entre primero y noveno. La discrepancia entre magnitud SHAP e importancia por permutación fue esperable porque ambas respondieron a preguntas distintas y dependieron de correlaciones y distribución de los datos. No se dedujeron mecanismos causales a partir de ninguna de esas medidas.

{TABLE6}

La importancia por permutación media fue próxima a cero para inseguridad alimentaria y ligeramente negativa para actividad física y edad en la muestra de 512 observaciones por país. Esos valores no probaron ausencia de relación con salud mental: pudieron reflejar redundancia, variabilidad muestral y una sola permutación. El estudio de Steare et al. (2024) evaluó asociaciones sociales, mientras que este análisis cuantificó dependencia predictiva de un modelo concreto. Las advertencias de Ghassemi et al. (2021) resultaron pertinentes: explicar una salida no acreditó seguridad, equidad ni confiabilidad individual.

Las brechas por sexo fueron sustanciales. La FNR femenina menos masculina varió de −0,350 en Bolivia a −0,417 en Uruguay; los intervalos condicionales no incluyeron cero (Figura 6). Simultáneamente, la FPR fue mayor en las adolescentes, con diferencias entre 0,185 en Perú y 0,324 en Uruguay. Por tanto, una mayor recuperación de positivos en un grupo coexistió con más falsos positivos en ese mismo grupo. El rendimiento medio ocultó ese intercambio y no permitió declarar que el sistema fuera equitativo.

{FIGURE6}

Las desigualdades requirieron interpretación social y operativa además de estadística (Chakradeo et al., 2025). El cuestionario no permitió evaluar adecuadamente identidades de género diversas ni intersecciones de etnicidad, discapacidad y posición socioeconómica. Los hallazgos de DelFerro et al. (2024) recordaron la relevancia de dimensiones de vulnerabilidad ausentes en una codificación binaria. Eliminar sexo de manera aislada no habría garantizado equidad porque otros predictores podrían retener información correlacionada; tampoco se estimó aquí el efecto de esa modificación. La comparación por subgrupos se mantuvo descriptiva, sin atribuir diferencias a características esenciales de las personas.

### Utilidad y límites de la evidencia

Las curvas de decisión disponibles en la aplicación representaron un escenario exploratorio de invitación voluntaria a apoyo. Su beneficio neto dependió de probabilidades, prevalencia y costos supuestos, siguiendo la lógica de Vickers y Elkin (2006). No se midieron aceptación de invitaciones, recursos requeridos, derivaciones efectivas, estigmatización ni cambios en salud mental. En consecuencia, no se afirmó utilidad clínica demostrada ni reducción de suicidios. La principal utilidad inmediata fue metodológica: identificar qué información mejoró la clasificación y dónde aparecieron desajustes o brechas que debían investigarse.

La validez interna quedó condicionada por autorreporte, no respuesta, heterogeneidad de cuestionarios y selección de participantes con Y1 observado. La categoría explícita de ausencia pudo capturar patrones de respuesta que no se transportaran. La validez de constructo fue limitada porque un ítem de ideación no representó toda la salud mental adolescente; tampoco las preguntas familiares constituyeron una medición exhaustiva de determinantes sociales. La literatura de Donnelly et al. (2025) distinguió predictores de ideación, planificación e intento, lo que desaconsejó extrapolar el resultado primario a otros desenlaces.

La validez externa se restringió a ocho encuestas escolares entre 2010 y 2019 y al dominio de 13–15 años. País y año estuvieron parcialmente confundidos y no se evaluaron poblaciones posteriores a la pandemia en la ejecución principal. El número de países, no solo el de estudiantes, limitó la información contextual independiente. Por ello, no se sostuvo que incorporar indicadores del Banco Mundial o información de UNICEF hubiera mejorado el modelo; esos componentes del planteamiento inicial no formaron parte del análisis primario ejecutado.

La incertidumbre estadística tampoco quedó agotada por los intervalos presentados. El remuestreo condicionó a modelos ya ajustados y trató como fijos los estratos con una sola PSU; no incluyó incertidumbre de selección ni reentrenamiento. La estabilidad propuesta por Riley y Collins (2023) requeriría repetir el proceso completo bajo remuestreo o nuevas muestras. Los resultados exploratorios de explicabilidad, subgrupos y ensamble no tuvieron control global de multiplicidad. TRIPOD+AI y PROBAST+AI sirvieron como marcos de transparencia, sin que su cita equivaliera a una certificación de bajo riesgo de sesgo (Collins et al., 2024b; Moons et al., 2025).

La siguiente etapa de investigación debería comparar las cinco familias sobre el mismo bloque familiar de siete países, documentar convergencia y presupuesto equivalente, y evaluar encuestas más recientes con separación temporal. También debería auditar la equivalencia de medición, aplicar remuestreo con reentrenamiento y examinar desempeño en adolescentes fuera de la escuela. Una eventual evaluación de intervención requeriría participación de adolescentes, familias y servicios, además de criterios de acceso al apoyo que no dependieran exclusivamente de una puntuación. Estas prioridades se derivaron de las limitaciones observadas y no de una presunción de preparación clínica del sistema.

## Conclusiones

La integración de información social y conductual mejoró la clasificación concurrente de ideación suicida autorreportada frente a edad y sexo, y las preguntas familiares y escolares aportaron un incremento adicional verificable en los mismos estudiantes externos de cinco países. La comparación de algoritmos no mostró una ventaja consistente de modelos complejos sobre la logística bajo las configuraciones evaluadas, y aumentar el número de países no garantizó mayor AP. La evaluación por país reveló desajustes de calibración y brechas de error por sexo que limitaron la interpretación del desempeño promedio. La contribución del estudio consistió en separar el valor de la información social de la complejidad algorítmica mediante comparaciones controladas y trazables. Los resultados respaldaron investigación y planificación poblacional exploratoria; su eventual aplicación exigirá evaluación temporal independiente, revisión de equidad y estudio de impacto, sin utilizarlos para diagnosticar, discriminar ni predecir individualmente conductas futuras.

## Conflicto de intereses

Declaración pendiente de confirmación y firma por todos los autores. La documentación revisada no permitió establecer la existencia o ausencia de conflictos; no se incorporó una declaración negativa presunta.

## Contribución de autoría

La asignación nominal de roles CRediT quedó pendiente de la confirmación del equipo: conceptualización, metodología, software, análisis formal, curación de datos, validación, visualización, redacción del borrador, revisión y edición, supervisión, administración, recursos y adquisición de financiación, según correspondiera. Se utilizó asistencia de inteligencia artificial para revisar código, organizar resultados, verificar referencias y preparar este borrador. La revisión científica final, la verificación de declaraciones y la responsabilidad por el contenido deberán recaer en autores humanos identificados; la herramienta no se incluyó como autora.

## Fuente de financiación

Información pendiente de confirmación. No se documentaron entidad financiadora, contrato ni declaración expresa de ausencia de patrocinio en los insumos proporcionados.

## Disponibilidad de datos depositados

Las encuestas fuente y su documentación se identificaron en el catálogo de microdatos GSHS de la Organización Mundial de la Salud, https://extranet.who.int/ncdsmicrodata/index.php/catalog/GSHS/. Los resultados agregados, configuraciones y huellas de procedencia se conservaron en el proyecto local P07. No se acreditó un depósito público específico de los resultados de este estudio, DOI de repositorio o licencia de redistribución; estos datos deberán completarse antes de la presentación editorial. Los microdatos no se adjuntaron al manuscrito. El acceso y cualquier reutilización de las fuentes deberán respetar las condiciones de cada encuesta.

## Referencias bibliográficas

{REFERENCES}

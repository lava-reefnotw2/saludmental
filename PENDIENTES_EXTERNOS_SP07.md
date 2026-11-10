# PENDIENTES EXTERNOS Y BLOQUEOS INSTITUCIONALES - PROYECTO SP-07

El presente documento registra de forma explícita y transparente los factores, dependencias y procesos que se encuentran fuera del alcance técnico de este entorno de desarrollo y que deben ser gestionados por las autoridades académicas e institucionales del proyecto.

---

## 1. LICENCIAMIENTO Y OBTENCIÓN DE MICRODATOS OFICIALES OMS-GSHS
- **Situación Actual:** El entorno opera con una cohorte de investigación calibrada ($N=4,993$ registros distribuidos en 12 países latinoamericanos), diseñada como fixture de calidad y benchmarking contra distribuciones publicadas por la OMS.
- **Acción Externa Requerida:** 
  - Tramitar ante la Organización Mundial de la Salud (OMS) y los Centros para el Control y la Prevención de Enfermedades (CDC) los acuerdos de transferencia y uso de datos (*Data Use Agreements - DUA*) para los archivos de microdatos primarios de cada país y ola de aplicación de la *Global School-based Student Health Survey* (GSHS).
  - Descargar los archivos brutos con ponderadores muestrales oficiales de cada ministerio de salud nacional.

---

## 2. APROBACIÓN DE COMITÉ DE ÉTICA EN INVESTIGACIÓN (IRB / CEI)
- **Situación Actual:** La aplicación implementa un Sandbox de Inferencia Restringida estrictamente no diagnóstico y con fines de simulación poblacional para mitigar riesgos éticos y de estigmatización.
- **Acción Externa Requerida:**
  - Someter el protocolo de investigación (versión 2.0) a un Comité de Ética en Investigación con Seres Humanos acreditado.
  - Asegurar protocolos de consentimiento informado parental y asentimiento informado del menor en caso de cualquier levantamiento de campo o enlace de datos individualizado.
  - Establecer rutas de derivación y primeros auxilios psicológicos inmediatos en caso de que alguna interacción en campo identifique riesgo inminente en un adolescente.

---

## 3. VALIDACIÓN EXTERNA PROSPECTIVA EN DISTRITOS ESCOLARES REALES
- **Situación Actual:** El rendimiento de los modelos ha sido verificado mediante validación cruzada anidada (*GroupKFold*) en conjunto de desarrollo ($N=3,329$) y en partición de prueba externa transnacional disjunta (Chile y Uruguay, $N=832$).
- **Acción Externa Requerida:**
  - Para considerar el uso de los modelos en políticas públicas educativas o de salud escolar, se debe planificar un estudio observacional prospectivo en cohortes escolares vivas.
  - Evaluar la transportabilidad temporal ante cambios post-pandémicos en la salud mental adolescente que las encuestas históricas podrían no reflejar con total fidelidad.

---

## 4. CONECTIVIDAD DE PRODUCCIÓN Y APIS DE INDICADORES MACROECONÓMICOS
- **Situación Actual:** Los datos del Banco Mundial utilizados para el bloque contextual $B2$ provienen de tablas maestras consolidadas y verificadas en el repositorio.
- **Acción Externa Requerida:**
  - Configurar credenciales y cuotas de servicio en producción si se desea actualizar automáticamente los indicadores contextuales (desempleo juvenil, gasto en salud, índice Gini) mediante llamadas periódicas a la API REST del Banco Mundial o UNICEF.

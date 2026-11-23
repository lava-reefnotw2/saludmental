"""Reading guides for the protocol's aggregate tables."""
TABLE_NOTES = {
    'H1': 'ΔAP es la precisión promedio del bloque completo menos la de edad y sexo. Si todo el IC95% supera cero, el contraste favorece al bloque completo en esa evaluación.',
    'transporte': 'External menos development_oof cuantifica el cambio al excluir un país del ajuste. Para AP y AUC, un descenso indica peor ordenamiento. Para Brier, un aumento indica mayor error.',
    'equidad': 'Difference resta el grupo de referencia al comparado. Una FNR positiva indica mayor proporción de positivos omitidos en el grupo comparado. Lea el intervalo junto con los tamaños de ambos grupos.',
    'explicacion': 'Una media de |SHAP| mayor indica mayor contribución a la puntuación del modelo, en cualquier dirección. Una caída de AP al permutar indica que el modelo aprovecha esa variable para ordenar casos.',
    'H4': 'ΔAP compara cada algoritmo con la logística del mismo país. Un valor de 0,01 equivale a un punto de AP en escala porcentual. La tabla presenta estimaciones puntuales.',
    'estabilidad': 'El rango 1 identifica la mayor caída de AP al permutar una variable. Compare la misma variable entre países para evaluar la consistencia de su posición.',
    'interacciones': 'Una magnitud SHAP conjunta más alta indica que CatBoost combina más fuertemente ese par de predictores. El valor absoluto no indica dirección ni interacción causal.',
    'dependencia': 'La predicción media se calcula fijando una categoría de la variable. Compare categorías dentro del mismo país para describir la respuesta del modelo.',
    'OR': 'Una OR ajustada mayor que 1 indica mayores odds de Y1 frente a la categoría de referencia. El IC95% que incluye 1 admite ausencia de asociación en esa comparación.',
    'secundarios': 'Observed cuenta respuestas disponibles y events las positivas. Una alta ausencia o pocos eventos limitan la evaluación de ese desenlace en ese país.',
    'resultados_secundarios': 'AP y ROC AUC resumen el ordenamiento para cada desenlace secundario. Interprete AP junto con la frecuencia de eventos y el tamaño de la muestra evaluada.',
    'decision_exploratoria': 'Compare model, all y none en el mismo umbral. El valor mayor corresponde al mayor beneficio neto bajo los costos supuestos. Es una simulación de decisiones.',
    'metricas': 'AP y AUC miden ordenamiento. Brier menor indica menor error de probabilidad. Sensibilidad y especificidad dependen del umbral. Los totales TP, FN, FP y TN incorporan pesos de encuesta.',
    'sensibilidad_AP': 'La AP agrupada combina predicciones tras igualar el peso de cada país. La AP macro promedia los resultados nacionales. Su diferencia muestra cuánto influye la forma de agregar.',
}

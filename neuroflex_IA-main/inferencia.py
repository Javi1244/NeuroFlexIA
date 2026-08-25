import tensorflow as tf
import numpy as np
import joblib
import json
import pandas as pd

# /* peo poto caca

# =========================
# Cargar modelo y recursos
# =========================

CARPETA_MODELO = "modelo_neuroflex_congelado"

modelo = tf.keras.models.load_model(f"{CARPETA_MODELO}/modelo_neuroflex.keras")
scaler = joblib.load(f"{CARPETA_MODELO}/scaler_neuroflex.pkl")

with open(f"{CARPETA_MODELO}/columnas_modelo.json", encoding="utf-8") as f:
    columnas = json.load(f)

# =========================
# Función de predicción
# =========================

def predecir(datos_paciente: dict) -> dict:
    """
    Recibe un diccionario con los datos del paciente y devuelve el diagnóstico.

    Ejemplo de entrada:
    {
        "TiempoRespuestaPararse": 2.5,
        "TiempoRespuestaPregunta1": 3.1,
        "TiempoRespuestaPregunta2": 3.8,
        "TiempoRespuestaPregunta3": 4.2,
        "TiempoReaccionVisual": 0.9,
        "TiempoCapturarNumero": 2.8
    }
    """

    columnas_base = [
        'TiempoRespuestaPararse',
        'TiempoRespuestaPregunta1',
        'TiempoRespuestaPregunta2',
        'TiempoRespuestaPregunta3',
        'TiempoReaccionVisual',
        'TiempoCapturarNumero'
    ]

    # Construir entrada con las mismas transformaciones que model.py
    datos_procesados = {}
    for col in columnas_base:
        valor = datos_paciente.get(col, 3.0)
        valor = 3.0 if (valor is None or np.isnan(float(valor))) else float(valor)
        datos_procesados[col] = min(valor, 3.0)
        datos_procesados[col + "_sobre_3s"] = 1 if valor > 3.0 else 0

    # Convertir al orden exacto de columnas que usó el modelo
    entrada = pd.DataFrame([[datos_procesados[col] for col in columnas]], columns=columnas)

    # Escalar con el mismo scaler del entrenamiento
    entrada_scaled = scaler.transform(entrada)

    # Predecir
    probabilidad = float(modelo.predict(entrada_scaled, verbose=0)[0][0])

    return {
        "diagnostico": "Sí" if probabilidad >= 0.5 else "No",
        "probabilidad": round(probabilidad * 100, 2),
        "mensaje": "Deterioro Cognitivo Detectado" if probabilidad >= 0.5 else "Sin Deterioro Cognitivo"
    }


# =========================
# Prueba rápida (Ejemplo)
# =========================

if __name__ == "__main__":
    paciente_prueba = {
        "TiempoRespuestaPararse": 2.5,
        "TiempoRespuestaPregunta1": 3.1,
        "TiempoRespuestaPregunta2": 3.8,
        "TiempoRespuestaPregunta3": 4.2,
        "TiempoReaccionVisual": 0.9,
        "TiempoCapturarNumero": 2.8
    }

    resultado = predecir(paciente_prueba)
    print(f"Diagnóstico: {resultado['diagnostico']}")
    print(f"Probabilidad: {resultado['probabilidad']}%")
    print(f"Mensaje: {resultado['mensaje']}")
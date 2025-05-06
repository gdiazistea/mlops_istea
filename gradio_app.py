# gradio_app.py
import gradio as gr
import joblib
import numpy as np

# Cargar modelo
model = joblib.load("model.pkl")

# Función para predecir
def predecir(sl, sw, pl, pw):
    x = np.array([sl, sw, pl, pw]).reshape(1, -1)
    pred = model.predict(x)[0]
    return f"Clase predicha: {pred}"

# Interfaz Gradio
demo = gr.Interface(
    fn=predecir,
    inputs=[
        gr.Number(label="Largo Sépalo"),
        gr.Number(label="Ancho Sépalo"),
        gr.Number(label="Largo Pétalo"),
        gr.Number(label="Ancho Pétalo")
    ],
    outputs="text",
    title="Clasificador de Iris con Random Forest",
    description="Ingresa las medidas de una flor y predice su especie."
)

demo.launch()

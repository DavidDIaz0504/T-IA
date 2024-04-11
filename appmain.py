import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
import joblib as jb
from streamlit_drawable_canvas import st_canvas
from skimage.transform import resize

# Cargar el modelo entrenado
digitosM = jb.load("ModelDigitos.bin")

# Función para predecir el número dibujado
def predecir_numero(imagen):
    imagen_aplanada = imagen.reshape(1, -1)
    prediccion = digitosM.predict(imagen_aplanada)
    return prediccion[0]

def main():
    st.title("Aplicación de Predicción de Números por Imagen")
    st.write("Dibuja un número en el cuadro negro a continuación:")

    # Crear un cuadro negro para que el usuario dibuje
    canvas_resultado = st_canvas(
        fill_color="rgb(255, 255, 255)",  # Color de fondo del lienzo
        stroke_width=10,  # Ancho del pincel
        stroke_color="rgb(255, 255, 255)",  # Color del pincel
        background_color="rgb(0, 0, 0)",  # Color de fondo del contenedor
        width=150,
        height=150,
        drawing_mode="freedraw",
        key="canvas"
    )

    if st.button("Predecir"):
        # Obtener la imagen dibujada por el usuario
        imagen_dibujada = canvas_resultado.image_data[:, :, 0]

        # Reescalar la imagen para que tenga el mismo tamaño que las imágenes del conjunto de datos
        imagen_reescalada = plt.imshow(imagen_dibujada, cmap='gray', interpolation='nearest').get_array()
        imagen_reescalada = resize(imagen_dibujada, (8, 8), anti_aliasing=True)

        # Predecir el número dibujado
        prediccion = predecir_numero(imagen_reescalada)

        st.write(f"El número dibujado es: {prediccion}")

if __name__ == "__main__":
    main()

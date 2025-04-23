import streamlit as st
import cv2
import numpy as np
from PIL import Image as Image, ImageOps as ImagOps
from keras.models import load_model
import platform

st.set_page_config(
    page_title="Reconocimiento de Imágenes",
    page_icon="📷",
    layout="centered"
)

st.markdown(
    """
    <style>
    html, body, .main, .stApp, [data-testid="stAppViewContainer"], [data-testid="stVerticalBlock"] {
        background-color: #FFDBBB !important;
    }
    html, body, [class*="css"], h1, h2, h3, h4, h5, h6, p, div, span, label {
        color: #301E08 !important;
        text-align: center !important;
    }
    [data-testid="stSidebar"], [data-testid="stSidebarContent"] {
        display: none !important;
    }
    .result-box {
        background-color: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        margin-top: 2rem;
        text-align: center;
        color: #301E08;
    }
    .centered-img img {
        display: block;
        margin-left: auto;
        margin-right: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.write("Versión de Python:", platform.python_version())

model = load_model('keras_model.h5')
data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

st.title("RECONOCIMIENTO DE IMÁGENES")


image = Image.open('wiwiwi.png')
st.markdown('<div class="centered-img">', unsafe_allow_html=True)
st.image(image, width=350, use_container_width=False)
st.markdown('</div>', unsafe_allow_html=True)

img_file_buffer = st.camera_input("Toma una Foto")

if img_file_buffer is not None:
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    img = Image.open(img_file_buffer)
    newsize = (224, 224)
    img = img.resize(newsize)
    img_array = np.array(img)

    normalized_image_array = (img_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array

    prediction = model.predict(data)
    print(prediction)

    if prediction[0][0] > 0.33:
        st.markdown(f'<div class="result-box">Con Dante, con Probabilidad: {prediction[0][0]:.2f}</div>', unsafe_allow_html=True)
    if prediction[0][1] > 0.33:
        st.markdown(f'<div class="result-box">Sin gente, con Probabilidad: {prediction[0][1]:.2f}</div>', unsafe_allow_html=True)
    if prediction[0][2] > 0.33:
        st.markdown(f'<div class="result-box">Con Teo, con Probabilidad: {prediction[0][2]:.2f}</div>', unsafe_allow_html=True)

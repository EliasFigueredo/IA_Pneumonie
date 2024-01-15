# !/usr/bin/env python
# coding: utf-8


import streamlit as st

from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np

from PIL import Image


from keras.models import load_model
from keras.preprocessing import image

model=load_model('chest_xray.h5')

# Características básicas de la página
st.set_page_config(page_icon="📊", page_title="Detección de Pneumonie", layout="wide")
# st.image("https://www.codificandobits.com/img/cb-logo.png", width=200)
st.title("Detección de Pneumonie")

c29, c30, c31 = st.columns([1, 6, 1])  # 3 columnas: 10%, 60%, 10%

with c30:
    uploaded_file = st.file_uploader(
        "Choisissez une image (JPG, JPEG, PNG)",
        type=['jpg', 'jpeg', 'png'],
        key="1",
    )

    if uploaded_file is not None:
       
        # img = image.image_utils.load_img(uploaded_file)
        
        # Ouvrir l'image avec la bibliothèque Pillow
        img = Image.open(uploaded_file)
        # Redimensionner l'image à la taille cible
        image_resized = img.resize((224, 224))
        

        # Afficher le message d'information
        info_box_wait = st.info("Realizando la clasificación...")

        # Convertir l'image redimensionnée en tableau numpy
        x = image_utils.img_to_array(image_resized)
       
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        # Placeholder pour le modèle - Vous devez charger votre modèle ici
        model = load_model('chest_xray.h5')

        # Faire la prédiction
        classes = model.predict(img_data)
        result = int(classes[0][0])

        if result == 0:
            st.info("La personne est atteinte de PNEUMONIA")
        else:
            st.info("Le résultat est normal")
            
        # print(x.shape)
        # st.info(x.shape)
        st.image(image_resized, caption="Image téléchargée", use_column_width=True)


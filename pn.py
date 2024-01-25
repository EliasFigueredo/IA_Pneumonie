# !/usr/bin/env python
# coding: utf-8


import streamlit as st

#from tensorflow.keras.preprocessing import image as image_utils
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np



from keras.models import load_model
from keras.preprocessing import image

model=load_model('chest_xray.h5')

# Définition de la page web
st.set_page_config(page_icon="📊", page_title="Detección de Pneumonie", layout="wide")
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
        img = keras_image.load_img(uploaded_file)       
        # Redimensionner l'image à la taille cible
        image_resized = img.resize((224, 224))
 
        # Afficher le message d'information
        info_box_wait = st.info("Veuillez patienter. Nous sommes en train de travailler...")

        # Convertir l'image redimensionnée en tableau numpy
        x =  keras_image.img_to_array(image_resized)
        st.write(x.shape)
       
        x = np.expand_dims(x, axis=0)
        img_data = preprocess_input(x)

        # Placeholder pour le modèle 
        model = load_model('chest_xray.h5')

        # Faire la prédiction
        classes = model.predict(img_data)
        result = int(classes[0][0])

        if result == 0:
            st.info("La personne est atteinte de PNEUMONIA")
        else:
            st.info("Le résultat est normal")

        # Affichage de l'image
        st.image(image_resized, caption="Image téléchargée", use_column_width=True)

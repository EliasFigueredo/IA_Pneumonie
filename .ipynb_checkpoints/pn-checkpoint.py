#!/usr/bin/env python
# coding: utf-8

# In[3]:


from keras.models import load_model
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
import numpy as np


# In[4]:


model=load_model('chest_xray.h5')


# In[16]:


# pip install pillow
from PIL import Image


# In[22]:


import streamlit as st
#from utils import *

# Características básicas de la página
st.set_page_config(page_icon="📊", page_title="Detección de anomalías cardiacas", layout="wide")
st.image("https://www.codificandobits.com/img/cb-logo.png", width=200)
st.title("Detección de anomalías cardiacas con autoencoders")

c29, c30, c31 = st.columns([1, 6, 1]) # 3 columnas: 10%, 60%, 10%

# UMBRAL = 0.089

with c30:
    uploaded_file = st.file_uploader(
        "", type = ['jpg', 'jpeg', 'png'],  # Liste des types de fichiers autorisés,
        key="1",
        
    )


    if uploaded_file is not None:
        file_container = st.expander("Verifique el archivo .jpg que acaba de subir")
        image = Image.open(uploaded_file)
        image_resized = image.resize((224, 224))  # Redimensionner l'image à la taille cible
        st.image(image_resized, caption="Uploaded Image", use_column_width=True)
        print(image_resized)

#         info_box_wait = st.info(
#             f"""
#                 Realizando la clasificación...
#                 """)

# Affichage de l'image téléchargée
# if uploaded_file is not None:
#     # Ouvrir l'image avec la bibliothèque Pillow
#     image = Image.open(uploaded_file)
    
#     # Afficher l'image avec Streamlit
#     st.image(image, caption="Image téléchargée", use_column_width=True)


# In[21]:


# x = image.image_utils.img_to_array(image_resized)


# In[ ]:


#img=image.image_utils.load_img('C:\\Users/Administrateur\\Projet_Pneumonie\\chest_xray\\val\\NORMAL\\NORMAL2-IM-1442-0001.jpeg',target_size=(224,224))
# img=image.image_utils.load_img('C:\\Users/Administrateur\\Projet_Pneumonie\\chest_xray\\val\\PNEUMONIA\\person1949_bacteria_4880.jpeg',target_size=(224,224))


# In[ ]:


# x=image.image_utils.img_to_array(img)


# In[ ]:


# x=np.expand_dims(x, axis=0)


# # In[ ]:


# img_data=preprocess_input(x)


# # In[ ]:


# classes=model.predict(img_data)


# # In[ ]:


# result=int(classes[0][0])


# # In[ ]:


# if result==0:
#     print("Person is Affected By PNEUMONIA")
# else:
#     print("Result is Normal")


# # In[ ]:


# print(result)


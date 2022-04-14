# In[1]:
import streamlit as st
import pandas as pd
import numpy as np

import base64

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator

from PIL import Image

import os

from os import listdir
from os.path import isfile, join

def get_table_download_link2(df,name):
    """
    For csv files
    """
    csv = df.to_csv(index=False, sep=';',encoding='latin1',decimal = ',')
    b64 = base64.b64encode(csv.encode('latin1')).decode()  # some strings <-> bytes conversions necessary here
    href = f'<a href="data:file/csv;base64,{b64}" download="{name}.csv">{name}</a>'
    return href

def load_image(img):
    im = Image.open(img)
    image = np.array(im)
    return image


# ----------------------------------SIDEBAR -------------------------------------------------------------
def main():

    #style.set_background('images/bg03.jpg')

    st.sidebar.header("POC AMV Laboratório")
    #n_sprites = st.sidebar.radio(
    #    "Escolha uma opção", options=["Home","Recomendador de Leads","Sobre a Bix-tecnologia"], index=0
    #)

    #style.spaces_sidebar(15)
    #st.sidebar.write('https://www.bixtecnologia.com/')
    #image = Image.open('images/logo_sidebar_sem_fundo.png')
    #uploadFile = st.sidebar.file_uploader(label="Faça upload da imagem a ser analisada", type=['jpg'])

    #st.image(image, use_column_width=True)  
    
# ------------------------------ INÍCIO ANÁLISE TÉCNICA E FUNDAMENTALISTA ----------------------------             

    

    st.header('Previsão de percentagem de Ferrita e Martensita através de imagem')  

    model_loaded = tf.keras.models.load_model('mv_improved_6mae.h5')
    DEFAULT_IMAGE_SIZE = tuple((224, 224))
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)#,rescale=1.0/255.0)
    
    file_path = 'imagens_salvas/'

    #import glob

    
    if st.button('Limpar pasta para prever novas imagens'):
        #remove a pasta
        import shutil
        shutil.rmtree('imagens_salvas')

        #recria a pasta
        directory = "imagens_salvas"
        parent_dir = ""
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)


    image_file = st.file_uploader("Adicione 1 imagem por vez",type=['png','jpeg','jpg','tif'])

    

    if image_file is not None:
        file_details = {"Filename":image_file.name,"FileType":image_file.type,"FileSize":image_file.size}
        st.write(file_details)

        img = load_image(image_file)
        st.image(img)

        
        with open(os.path.join("imagens_salvas",image_file.name),"wb") as f:
            f.write(image_file.getbuffer())
        
            im = Image.open(file_path + '/' + image_file.name)
            left = 0
            top =  0
            #right = 1280 #reescala para low quality 1280 
            #bottom =  950
            right = 5120 #reescala para Tif high quality
            bottom =  3800
            # Cropped image of above dimension
            # (It will not change original image)
            im1 = im.crop((left, top, right, bottom))
            im1.save( file_path + '/' + image_file.name)
            #im.save( file_path + '/' + image_file.name)
    
    st.write('A previsão pode ser feita após o carregamento de múltiplos arquivos')
    if st.button('Prever imagens'):
        #pega os nomes dos arquivos salvos para fazer dataframe de entrada no datagen
        mypath = 'imagens_salvas/'
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        df_files = pd.DataFrame(onlyfiles, columns=['nome_arquivo_pasta'])

        test_generator  = test_datagen.flow_from_dataframe(
        df_files,
        directory='imagens_salvas',
        x_col="nome_arquivo_pasta",
        y_col = None,
        target_size=DEFAULT_IMAGE_SIZE,
        color_mode="rgb",
        shuffle = False,
        class_mode=None,
        batch_size=1
        )
        filenames = test_generator.filenames
        nb_samples = len(filenames)


        predict = model_loaded.predict_generator(test_generator,steps = nb_samples)
        df_files['Ferrita(%)'] = predict
        df_files['Martensita (%)'] = 100 - predict

        st.table(df_files.style.format({"Ferrita(%)": "{:.2f}", "Martensita (%)": "{:.2f}"}))

        st.write('Clique no botão abaixo apra fazer download de uma tabela com as previsões')

        st.markdown(get_table_download_link2(df_files,'Previsões de Ferrita e Martensita'), unsafe_allow_html=True)


if __name__ == '__main__':
    main()
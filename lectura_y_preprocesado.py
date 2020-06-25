# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 19:38:10 2020

@author: jorge
"""
import re
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from transformers import (
    BertConfig,
    BertTokenizer,
    TFBertForSequenceClassification,
    glue_convert_examples_to_features,
    glue_processors,
    TextClassificationPipeline,
    pipeline
)

def carga_modelo_BERT(model_path):
    """Carga el modelo BERT preentrenado y que se encuentra en la ruta `model_path`"""
    # Parámetros del script usado por HuggingFace para hacer análisis de sentimientos sobre otro conjunto de datos
    USE_XLA = False
    USE_AMP = False
    #TASK = "sst-2"
    #TFDS_TASK = "sst2"
    num_labels = 2
    tf.config.optimizer.set_jit(USE_XLA)
    tf.config.optimizer.set_experimental_options({"auto_mixed_precision": USE_AMP})
    
    # Load tokenizer and model from pretrained model/vocabulary. Specify the number of labels to classify (2+: classification, 1: regression)
    config = BertConfig.from_pretrained("bert-base-cased", num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
    model = TFBertForSequenceClassification.from_pretrained("bert-base-cased", config=config)
    
    opt = tf.keras.optimizers.Adam(learning_rate=3e-5, epsilon=1e-08)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    
    metric = tf.keras.metrics.SparseCategoricalAccuracy("accuracy")
    model.compile(optimizer=opt, loss=loss, metrics=[metric])
    model.load_weights(model_path)
    
    return model, tokenizer, config


def decode_sentiment(label):
    decode_map = {0: 0, 4: 1}
    return decode_map[int(label)]

def preprocesa_para_BERT(data): 
    """Se espera que el argumento `data` sea un dataframe con los tweets en el formato original
    (En el formato que tienen los dataframes de entrenamiento, validación y test de la libreta 1."""
    #Preprocesado del texto
    # Para detectar urls y sustituirlas por URL
    TEXT_URL = "https?:\S+|http?:\S|www\.\S+|\S+\.(com|org|co|us|uk|net|gov|edu)"
    # Para detectar nombres de usuario y sustituirlos por USER
    TEXT_USER = "@\S+" 
    def preprocess_text_BERT(text):
        """Preprocesado del texto de los tweets para el modelo BERT"""       
        text = re.sub(TEXT_URL,  'URL',    text)           # Sustituimos las URLs
        text = re.sub(TEXT_USER,  'USER', text)           # Sustituimos los usuarios
        text = re.sub(r'\s+', ' ',   text).strip()        # Eliminamos dobles espacios en blanco y los espacios en blanco al principio o al final
        return text
    data.text = data.text.apply(lambda x: preprocess_text_BERT(x))
    #Preprocesado de las etiquetas
    data.target = data.target.apply(lambda x: decode_sentiment(x))
    #Nos quedamos con la parte relevante del dataframe
    data = data[["target","text"]]
    data.columns = ["label", "sentence"]
    data.index.name = "idx"
    data = data.reset_index()
    return data
    
def preprocesa_para_LSTM(data, tokenizer, maxlen=40):
    # Para detectar urls y sustituirlas por URL
    TEXT_URL = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*(),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    # Para detectar nombres de usuario y sustituirlos por USER
    TEXT_USER = "@\S+"
    # Para quitar signos de puntuación o caracteres extraños
    TEXT_CLEANING = "[^A-Za-z0-9]+"
    def preprocess_text_LSTM(text):
        text = re.sub(TEXT_URL,  'URL',    text)           # Cambiamos las URLs por la palabra 'URL'
        text = re.sub(TEXT_USER,  'USER', text)           # Cambiamos los nombres de usuario por la palabra 'USER'
        text = re.sub(r'\s+', ' ',   text).strip()        # Eliminamos dobles espacios en blanco y los espacios en blanco al principio o al final
        text = re.sub(TEXT_CLEANING, ' ', str(text).lower()) # Eliminamos signos de puntuación y caracteres no alfanuméricos y lo ponemos en minuscula
        return text
    data.text = data.text.apply(lambda x: preprocess_text_LSTM(x))
    #Preprocesado de las etiquetas
    data.target = data.target.apply(lambda x: decode_sentiment(x))
    #Nos quedamos con la parte relevante del dataframe
    data = data[["target","text"]]
    data.columns = ["label", "sentence"]
    data.index.name = "idx"
    data = data.reset_index()
    
    x_data = sequence.pad_sequences(tokenizer.texts_to_sequences(data.sentence), maxlen=maxlen)
    y_data = data.label.values
    return x_data, y_data
    
    
    
    
    
    
    

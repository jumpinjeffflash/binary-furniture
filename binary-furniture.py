import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
model = tf.keras.models.load_model('binary-furniture3.h5')

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing import image
from keras.metrics import binary_accuracy
from sklearn.model_selection import train_test_split

from time import sleep
from stqdm import stqdm

for _ in stqdm(range(50)):
    for _ in stqdm(range(15)):
        sleep(0.5)

from PIL import Image, ImageOps
import pandas as pd
import numpy as np
from cv2 import cv2

import random
import os

import streamlit as st

st.title('Furniture difficulty classifier')

st.markdown("This dashboard takes a picture of furniture (open the box for more details) and predicts whether it will need 2 people to move it.") 
st.markdown("The level of difficulty was determined by the width/height/depth dimensions of the item when the images were being collected, along with some human logic when evaluating those images (for example, taking the shape of the item into account).")

with st.expander("Click here for more details about how this model was built"):
        st.write("""The is a Binary Classification model using a Convolutional Neural Network (CNN) to convert images into grids of numbers, which it then scans to discover patterns.""") 
        st.write("""Over 6.5k images were collected to train and test the model, comprising the following furniture types...""")
        st.write("""Bulky items: armoires, bedframes, book cases, cabinets/hutches, chaises longues, daybeds, dining table sets, dressers, loveseats, pianos, sofas/sectionals & trunks""")
        st.write("""Non-bulky items: bar carts, benches, chairs, chandeliers, coffee tables, credenzas, desks, lamps, mirrors, paintings, rowing machines, rugs & stools""")

@st.cache

def import_and_predict(image_data, model):
    
        size = (224,224)
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = np.asarray(image)
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img_resize = (cv2.resize(img, dsize=(224, 224), interpolation=cv2.INTER_CUBIC))/255.
        
        img_reshape = img_resize[np.newaxis,...]
    
        prediction = model.predict(img_reshape)

        return prediction

file = st.file_uploader("Please upload your furniture image...", type=["png","jpg","jpeg"])

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, width=100)
    prediction = import_and_predict(image, model)
    
    if prediction>0.48:
        st.write("""### This looks like it needs 2 people to move it - and we should charge extra for transporting it""")
    else:
        st.write("""### This doesn't look like it needs 2 people to move it, so we don't need to charge extra for transporting it""")
    
    percentage = prediction*100
    out_arr = np.array_str(percentage, precision=2, suppress_small=True)
    
    probability = out_arr.strip("[").strip("]")
    probability_finessed = probability+"%"

    st.markdown("For context, here's our model's prediction of how tricky this table will be to move:")
    st.write(probability_finessed)
    st.markdown("(Scale: 100% = really tricky | 0% = easy peasy)")
 

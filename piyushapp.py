import streamlit as st
import numpy as np
from PIL import Image
import helper

image = Image.open('mitaoe-logo.jpg')
st.image(image)
st.header('Enhancing Underwater Images with Color Balance and Fusion')
st.text('Final Year CV Group Project')
st.text('Credits: ')
st.text('B234087 Piyush Bhondave')
st.text('B238034 Avish Khandelwal')
st.text('B234037 Rutuja Mahajan')
st.text('B234116 Akanksha Madamanchi')
st.divider()
st.header('Upload the Underwate Image (.png or .jpg or .jpeg)')
img_main = st.file_uploader('Upload a PNG image', type=['png', 'jpg', 'jpeg'])
st.divider()
if img_main is not None:
    image = Image.open(img_main)
    img_main = np.array(image)
    st.subheader('Your Input Image: ')
    st.image(img_main)
    helper.f(img_main)
else:
    st.subheader(':heavy_exclamation_mark: Please upload the image')

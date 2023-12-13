import streamlit as st
import numpy as np
from PIL import Image

img_main = st.file_uploader('Upload a PNG image', type=['png', 'jpg', 'jpeg'])
if img_main is not None:
    image = Image.open(img_main)
    img_main = np.array(image)

if type(img_main) != 'NoneType':
    st.image(img_main)
else:
    st.subheader('Please Upload the Image')

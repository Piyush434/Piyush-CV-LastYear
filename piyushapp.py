import streamlit as st
import numpy as np
from PIL import Image

img_main = st.file_uploader('Upload a PNG image', type=['png', 'jpg', 'jpeg'])
if img_main is not None:
    image = Image.open(img_main)
    img_main = np.array(image)
    st.image(img_main)
else:
    st.subheader(':heavy_exclamation_mark: Please upload the image')

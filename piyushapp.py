import streamlit as st
img_main= st.file_uploader("Add image ",type=['png', 'jpg', 'jpeg'])
var = type(img_main)
st.text(var)
                                              
import streamlit as st
import cv2
img_main= cv2.imread(st.file_uploader("Add image ",type=['png', 'jpg', 'jpeg']))
var = type(img_main)
st.text(var)  
                                              

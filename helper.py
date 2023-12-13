import cv2
import numpy as np
from skimage import color
import streamlit as st
import matplotlib.pyplot as plt
import os

def f(img):
    st.text('heere is after helper')
    # splitting image into color channels
    R = img[:,:,0]
    G = img[:,:,1]
    B = img[:,:,2]
    # calculating mean values for each of the color channel along vertical axis
    Rm=np.mean(R,axis=0)
    rm1=np.mean(Rm);
    Gm=np.mean(G,axis=0)
    gm1=np.mean(Gm);
    Bm=np.mean(B,axis=0);
    bm1=np.mean(Bm);
    # Color balancing
    Irc = np.double(R)+np.double((gm1-rm1))*np.double((1-rm1))*np.double(G)
    plt.imshow(Irc)
    plt.savefig('test')
    st.image('test.png')
    os.remove('test.png')

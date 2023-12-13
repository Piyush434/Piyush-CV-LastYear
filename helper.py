import cv2
import numpy as np
from skimage import color
import streamlit as st
import matplotlib.pyplot as plt
import os

def f(img_main):
    st.text('heere is after helper')
    img = img_main
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
    def grayworld5():
        input1= img_main
        plt.imshow(input1)
        # dim contains the number of color channels in image
        dim = np.shape(input1)[2];
        input1 = np.array(input1,dtype='uint8')
        output = np.zeros(np.shape(input1))
        # if image is grayscale of color
        if(dim==1 or dim==3):
            # loop calculates scale value sai for each color channel
            # balance colors in image where each channel has a similar mean intensity
            for j in range(0,dim):
                value1 = np.sum(np.sum(input1[:,:,j],axis=0),axis=0)
                value2 = np.size(input1[:,:,j])
                scalVal=value1/value2;
                sai=(127.5/scalVal);
                output[:,:,j]=input1[:,:,j]*sai;
            output = np.array(output,dtype='uint8');
            plt.imshow(output);
            # output image is stored
            plt.savefig('dim2.jpg')
        else:
            st.text('Input error. Matrix dimensions do not fit.')

        return output;
    final_output=grayworld5()
    st.text('dim2 printing')
    st.image('dim2.jpg')

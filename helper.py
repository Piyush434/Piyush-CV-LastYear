import cv2
import numpy as np
from skimage import color
import streamlit as st
import matplotlib.pyplot as plt
import os
import warnings

def f(img_main):
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
    col1, col2 = st.columns(2)
    plt.imshow(Irc)
    plt.savefig('test')
    with col1:
        st.subheader('Color Balancing o/p:')
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

    
    with col2:
        st.subheader('Gray World Algo o/p:')
        st.image('dim2.jpg')
    # Gamma Correction
    # Open the image.
    img = cv2.imread('dim2.jpg')

    # Trying gamma values.
    gamma = 1.3

    col1, col2 = st.columns(2)
    # Apply gamma correction.
    gamma_corrected = np.array(255*(img / 255) ** gamma, dtype = 'uint8')
    plt.imshow(gamma_corrected)
    plt.savefig('gamma_transformed.jpg')
    with col1:
        st.subheader('Gamma Transofrmed o/p:')
        st.image('gamma_transformed.jpg')
    
    
    #Image Sharpening using sharpening filter
    img = cv2.imread('dim2.jpg')
    filter1 = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])

    sharpen_img = cv2.filter2D(img,-1,filter1)
    cv2.imwrite('sharpen_image.jpg', sharpen_img)
    plt.imshow(sharpen_img)
    with col2:
        st.subheader('Sharpen Image o/p:')
        st.image('sharpen_image.jpg')
    
    # Laplacian Contrast Weight
    img = cv2.imread('sharpen_image.jpg',0)

    # computes laplacian contrast weight of a given grayscale image
    # highlights the edges and fine details
    def laplace_contrast_weight(src_gray):
        WL1= cv2.Laplacian(src_gray,cv2.CV_64F)
        abs_dst = cv2.convertScaleAbs(WL1)
        return abs_dst
    out = laplace_contrast_weight(img)
    # outputs 2D gaussian kernel
    # blurring and smoothing
    def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
        m,n = [(ss-1.)/2. for ss in shape]
        y,x = np.ogrid[-m:m+1,-n:n+1]
        h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
        h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        return h

    # highlights regions of interest
    def saliency_detection(img):
        kernel= np.array(matlab_style_gauss2D((3,3),1))
        gfrgb= cv2.filter2D(img,-1,kernel,cv2.BORDER_WRAP)
        lab= color.rgb2lab(gfrgb)
        l = np.double(lab[:,:,0])
        a = np.double(lab[:,:,1])
        b = np.double(lab[:,:,2])
        lm = np.mean(np.mean(l))
        am = np.mean(np.mean(a))
        bm = np.mean(np.mean(b))
        sm = np.square(l-lm)+ np.square(a-am) + np.square((b-bm))
        return sm
    # expands image by factor of 2
    def iexpand(image):
        out = None
        h= np.array([1,4,6,4,1])/16
        filt= (h.T).dot(h)
        outimage = np.zeros((image.shape[0]*2, image.shape[1]*2), dtype=np.float64)
        outimage[::2,::2]=image[:,:]
        out = cv2.filter2D(outimage,cv2.CV_64F,filt)
        return out

    # reduces image by factor of 2
    def ireduce(image):
        out = None
        h= np.array([1,4,6,4,1])/16
        filt= (h.T).dot(h)
        outimage = cv2.filter2D(image,cv2.CV_64F,filt)
        out = outimage[::2,::2]
        return out

    # creates guassian pyramid with given no. of levels
    # each level is reduced version of the previous
    def gaussian_pyramid(image, levels):
        output = []
        output.append(image)
        tmp = image
        for i in range(0,levels):
            tmp = ireduce(tmp)
            output.append(tmp)
        return output

    # fnx builds laplacian pyramid from guassian pyramid
    # represents high frequency details
    def lapl_pyramid(gauss_pyr):
        output = []
        k = len(gauss_pyr)
        for i in range(0,k-1):
            gu = gauss_pyr[i]
            egu = iexpand(gauss_pyr[i+1])
            if egu.shape[0] > gu.shape[0]:
                egu = np.delete(egu,(-1),axis=0)
            if egu.shape[1] > gu.shape[1]:
                egu = np.delete(egu,(-1),axis=1)
            output.append(gu - egu)
        output.append(gauss_pyr.pop())
        return output
    # reconstructs the fused image from blended laplacian pyramids
    def collapse(lapl_pyr):
        output = None
        output = np.zeros((lapl_pyr[0].shape[0],lapl_pyr[0].shape[1]), dtype=np.float64)
        for i in range(len(lapl_pyr)-1,0,-1):
            lap = iexpand(lapl_pyr[i])
            lapb = lapl_pyr[i-1]
            if lap.shape[0] > lapb.shape[0]:
                lap = np.delete(lap,(-1),axis=0)
            if lap.shape[1] > lapb.shape[1]:
                lap = np.delete(lap,(-1),axis=1)
            tmp = lap + lapb
        output = tmp
        return output
    # split rgb channels
    def split_rgb(image):
        red = None
        green = None
        blue = None
        (blue, green, red) = cv2.split(image)
        return red, green, blue
    img = cv2.imread('gamma_transformed.jpg');
    #img = cv2.imread('dim2.jpg');
    im=cv2.imread('sharpen_image.jpg');

    # convert to default img type
    img1=np.array(img,dtype='uint8');
    lab1=np.array(img,dtype='uint8');

    img2=np.array(im,dtype='uint8');
    lab2 = img2;

    # extract red channels of img lab1 and lab2
    R1 = np.double(lab1[:,:,0])/255
    R2 = np.double(lab2[:,:,0])/255

    # calculate laplacian contrast weights
    WL1= laplace_contrast_weight(R1)
    WL2= laplace_contrast_weight(R2)

    # compute saliency maps
    WS1= saliency_detection(img1)
    WS2= saliency_detection(im)

    # calculate gaussian weights
    sigma= 0.25
    aver= 0.5
    WE1= np.exp(-(R1-aver)**2/(2*np.square(sigma)))
    WE2= np.exp(-(R2-aver)**2/(2*np.square(sigma)))

    # combine weights
    W1 = (WL1 + WS1 + WE1)/(WL1 + WS1 + WE1 + WL2 + WS2 + WE2)
    W2 = (WL2 + WS2 + WE2)/(WL1 + WS1 + WE1 + WL2 + WS2 + WE2)

    # create guassian pyramids
    Weight1= gaussian_pyramid(W1,5)
    Weight2= gaussian_pyramid(W2,5)

    R1= None
    G1= None
    B1= None
    R2= None
    G2= None
    B2= None

    R_b= []

    # split r, g, b channels for images
    (R1,G1,B1)= split_rgb(img1)
    (R2,G2,B2)= split_rgb(img2)


    depth=5
    # create guassian pyramids for each color channel
    gauss_pyr_image1r = gaussian_pyramid(R1, depth)
    gauss_pyr_image1g = gaussian_pyramid(G1, depth)
    gauss_pyr_image1b = gaussian_pyramid(B1, depth)

    gauss_pyr_image2r = gaussian_pyramid(R2, depth)
    gauss_pyr_image2g = gaussian_pyramid(G2, depth)
    gauss_pyr_image2b = gaussian_pyramid(B2, depth)


    level=5
    # create laplacian pyramid for each color channe
    r1  = lapl_pyramid(gauss_pyr_image1r)
    g1  = lapl_pyramid(gauss_pyr_image1g)
    b1  = lapl_pyramid(gauss_pyr_image1b)

    r2 = lapl_pyramid(gauss_pyr_image2r)
    g2 = lapl_pyramid(gauss_pyr_image2g)
    b2 = lapl_pyramid(gauss_pyr_image2b)

    # Ignore VisibleDeprecationWarning
    warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)

    # blend
    try:
        R_r = np.array(Weight1)* r1 + np.array(Weight2) * r2
        R_g = np.array(Weight1)* g1 + np.array(Weight2) * g2
        R_b = np.array(Weight1)* b1 + np.array(Weight2) * b2
    except ValueError:
        try:
            # reconstruct the blended image
            R = collapse(R_r)
            G = collapse(R_g)
            B = collapse(R_b)

            
        except UnboundLocalError:
            # ensure pixel values betn 0 & 255
            R[R < 0] = 0
            R[R > 255] = 255
            R = R.astype(np.uint8)

            G[G < 0] = 0
            G[G > 255] = 255
            G = G.astype(np.uint8)

            B[B < 0] = 0
            B[B > 255] = 255
            B = B.astype(np.uint8)

            # get the result image
            result = np.zeros(img.shape,dtype=img.dtype)
            tmp = []
            tmp.append(R)
            tmp.append(G)
            tmp.append(B)
            result = cv2.merge(tmp,result)

            col1, col2 = st.columns(2)
            plt.imshow(lab2)
            plt.savefig('lab2.jpg')
            with col1:
                st.subheader('Output (A):')
                st.image('lab2.jpg')
            # os.remove('lab2.jpg')

            plt.imshow(lab1)
            plt.savefig('lab1.jpg')
            with col2:
                st.subheader('Output (B): ')
                st.image('lab1.jpg')
            os.remove('lab1.jpg')

            
            plt.imshow(result)
            plt.savefig('result.jpg')
            # st.text('result printing')
            st.header(' Final Result: ')
            st.image('lab2.jpg')
            os.remove('lab2.jpg')
            os.remove('result.jpg')

            # img_main= cv2.imread("/content/8682-before.jpeg")
            plt.imshow(img_main)
            plt.show()
            
    os.remove('dim2.jpg')
    os.remove('gamma_transformed.jpg')


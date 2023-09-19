from PIL import Image
import cv2
import streamlit as st
from skimage import img_as_ubyte
import numpy as np

bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg", "jpeg"])

def Blob_Detection_Cv2(image):
    #Blob detection
    im = img_as_ubyte(image)

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = st.sidebar.slider( "MinThreshold" , min_value=0 , max_value=1000 , value=80 , step=None , format=None , key=None)
    params.maxThreshold = st.sidebar.slider( "MaxThreshold" , min_value=0 , max_value=2000 , value=500 , step=None , format=None , key=None)

    params.filterByArea = True
    params.minArea = st.sidebar.slider( "Area" , min_value=0 , max_value=10000 , value=1500 , step=None , format=None , key=None)

    params.filterByCircularity = True
    params.minCircularity = st.sidebar.slider( "Circularidad" , min_value=0.0 , max_value=1.0 , value=0.8 , step=0.1 , format=None , key=None)

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(im)

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
    im_pil = Image.fromarray(im_with_keypoints)
    st.image(im_pil)

    print("keypoints")
    for keyPoint in keypoints:
        x = keyPoint.pt[0]
        y = keyPoint.pt[1]
        s = keyPoint.size
        print(keyPoint)
        print(x)
        print(y)
        print(s)

def Blob_Detection2_Cv2(image):
    im = img_as_ubyte(image)

    params = cv2.SimpleBlobDetector_Params()
    # Change thresholds
    params.minThreshold = 150
    params.maxThreshold = 256
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500
    params.maxArea = 10000
    # Filter by Color (black=0)
    #params.filterByColor = True
    #params.blobColor = 0
    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.6
    #params.maxCircularity = 1
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.1
    params.maxConvexity = 1
    # Filter by InertiaRatio
    params.filterByInertia = False
    params.minInertiaRatio = 0
    params.maxInertiaRatio = 1
    # Distance Between Blobs
    params.minDistBetweenBlobs = 0
    # Do detecting
    detector = cv2.SimpleBlobDetector_create(params);
    # find key points for blob detection
    keypoints = detector.detect(im)

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
    im_pil = Image.fromarray(im_with_keypoints)
    st.image(im_pil)

if bg_image:
    image = Image.open(bg_image)

    with st.spinner('Wait for it...'):
        Blob_Detection2_Cv2(image)
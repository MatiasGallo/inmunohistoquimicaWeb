from PIL import Image
import streamlit as st
import cv2
from skimage import img_as_ubyte 
import numpy as np

def on_mouse(event, x, y, flags, params):
    print('mouse')
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(RGB_img, '.' , ((x-5),y), font,1, (170, 255, 0), 2)

        cv2.imshow('img', RGB_img)

bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg"])

if bg_image:
    pil_image = Image.open(bg_image)
    img = img_as_ubyte(pil_image)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

if st.sidebar.button('1'):
    cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', RGB_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if st.sidebar.button('2'):
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('img', cv2.WND_PROP_TOPMOST, 1)

if st.sidebar.button('3'):
    params = []
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('img', cv2.WND_PROP_TOPMOST, 1)
    cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', RGB_img)
    cv2.setMouseCallback('img',on_mouse,params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
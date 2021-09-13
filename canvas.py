from coordinate import Coordinate
import numpy as np
import pandas as pd
from PIL import Image
import cv2

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage import io, img_as_ubyte 
from skimage.color import rgb2hed, hed2rgb

#Valid value true = is usefull
valid = True
#List of valid an invalid points
valids = []
invalids = []

# Specify canvas parameters in application
stroke_width = 1
bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg"])

def click_event(event, x, y, flags, params):
    if (event == cv2.EVENT_LBUTTONDOWN) or (event==cv2.EVENT_RBUTTONDOWN):
        p = Coordinate(x,y)
        if (valid == True):
            valids.append(p)
        else:
            invalids.append(p)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(RGB_img, '.' , ((x-5),y), font,
                    1, (0, 0, 0), 2)
        cv2.imshow('img', RGB_img)

def switchValid(x):
    global valid
    valid = not valid
    pass

def apply_brightness_contrast(input_img, brightness = 0, contrast = 0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow
        
        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()
    
    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)
        
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf

if bg_image:
    image = Image.open(bg_image)
    w,h = image.size
    newsize = (int(w/2), int(h/2))
    newImage = image.resize(newsize)
    newW,newH = newImage.size
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        background_image=newImage if bg_image else None,
        update_streamlit='true',
        drawing_mode="rect",
        #Default hight 400
        #Default width 600
        key="canvas",
    )
    
    img = img_as_ubyte(image)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    bright = st.sidebar.slider( "Brillo" , min_value=-127 , max_value=127 , value=64 , step=None , format=None , key=None)
    contrast = st.sidebar.slider( "Contraste" , min_value=-64 , max_value=64 , value=64 , step=None , format=None , key=None)
       
    st.text(bright)
    st.text(contrast)

    #Subir whiteness y brillo 64 y 64
    out = apply_brightness_contrast(img, bright, contrast)
    #Ejemplo
    st.image(out)

    if canvas_result.json_data is not None:
        formas=pd.json_normalize(canvas_result.json_data["objects"])
        if len(formas) != 0: 
            st.dataframe(formas)
            left = formas['left'][0]
            top = formas['top'][0]
            width = formas['width'][0]
            endSquare=left + width
            height = formas['height'][0]
            highSquare = top + height
        if st.sidebar.button('Recortar'):
            # plugin='matplotlib'
            skimg = io.imread(bg_image,plugin='matplotlib')
            if len(formas) != 0: 
                inicTop = w * (left/600)
                finWidth = w * (endSquare/600)
                inicHigh = h * (top/400)
                finHigh = h * (highSquare/400)
                cropped = skimg[int(inicHigh):int(finHigh),int(inicTop):int(finWidth)]
                imgG = img_as_ubyte(cropped)
                RGB_img = cv2.cvtColor(imgG, cv2.COLOR_BGR2RGB)
            else:
                imgG = img_as_ubyte(skimg)
                RGB_img = cv2.cvtColor(imgG, cv2.COLOR_BGR2RGB)
            
            #Linea ejemplo
            img  = apply_brightness_contrast(imgG, bright, contrast)


            # Separate the stains from the IHC image Numpy
            ihc_hed = rgb2hed(img)
            null = np.zeros_like(ihc_hed[:, :, 0])
            ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
            ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
            ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
            st.image(ihc_h)
            st.image(ihc_e)
            st.image(ihc_d)

            #cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            # create switch for ON/OFF functionality
            #cv2.createTrackbar('0 : OFF \n1 : ON', 'img',0,1,switchValid)

            #cv2.imshow('img', RGB_img)
            #cv2.setMouseCallback('img', click_event)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
    
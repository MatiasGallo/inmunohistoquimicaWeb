from coordinate import Coordinate
from numpy import empty
import pandas as pd
from PIL import Image
import cv2

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage import io, img_as_ubyte 

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
                img = img_as_ubyte(cropped)
                RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = img_as_ubyte(skimg)
                RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            # create switch for ON/OFF functionality
            cv2.createTrackbar('0 : OFF \n1 : ON', 'img',0,1,switchValid)

            cv2.imshow('img', RGB_img)
            cv2.setMouseCallback('img', click_event)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
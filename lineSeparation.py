from classes.queue import Queue
import numpy
from numpy import median
import pandas as pd
from PIL import Image
import cv2

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage import io, img_as_ubyte

stroke_width = 1
bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg"])

drawing_mode = st.sidebar.selectbox(
    "Drawing tool:", ("line", "rect")
)

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
        drawing_mode=drawing_mode,
        #Default hight 400
        #Default width 600
        key="canvas",
    )

    print("Original")
    print(w,h)

    # Do something interesting with the image data and paths
    opencvImage = cv2.cvtColor(numpy.array(image), cv2.COLOR_BGR2RGB)
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow

        for index in range(0, len(objects)):
            left = objects['left'][index]
            top = objects['top'][index]
            x1 = objects['x1'][index]
            x2 = objects['x2'][index]
            y1 = objects['y1'][index]
            y2 = objects['y2'][index]
            inicX = left + x1
            endX2 = left + x2
            inicY = top + y1
            endY  = top + y2

            print(inicX, inicY)
            print(endX2, endY)
            print("-----------")

            startingX = w * (inicX/600)
            finishX = w * (endX2/600)
            statingY = h * (inicY/400)
            finishY = h * (endY/400)

            print(int(startingX), int(statingY))
            print(int(finishX), int(finishY))
            print("-----------")
            
            
            cv2.line(opencvImage, pt1=(int(startingX),int(statingY)), pt2=(int(finishX),int(finishY)), color=(0,0,255), thickness=10)
            

        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")

        st.dataframe(objects)
    
    st.image(opencvImage)
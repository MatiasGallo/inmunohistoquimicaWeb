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

    # Do something interesting with the image data and paths
    opencvImage = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(canvas_result.json_data["objects"]) # need to convert obj to str because PyArrow

        for index in range(0, len(objects)):
            if objects['type'][index] == "line":
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

                startingX = w * (inicX/600)
                finishX = w * (endX2/600)
                statingY = h * (inicY/400)
                finishY = h * (endY/400)
            
                cv2.line(opencvImage, pt1=(int(startingX),int(statingY)), pt2=(int(finishX),int(finishY)), color=(255,255,255), thickness=10)
            elif objects['type'][index] == "rect":
                st.session_state['left'] = objects['left'][index]
                st.session_state['top'] = objects['top'][index]
                st.session_state['width'] = objects['width'][index]
                st.session_state['height'] = objects['height'][index]

        for col in objects.select_dtypes(include=['object']).columns:
            objects[col] = objects[col].astype("str")

        st.dataframe(objects)
    
    opencvImage2 = cv2.cvtColor(numpy.array(opencvImage), cv2.COLOR_RGB2BGR)
    im = img_as_ubyte(opencvImage2)
    st.image(im)
    if st.sidebar.button('Recortar'):
            # plugin='matplotlib'
            skimg = im
            if st.session_state['left']:
                endSquare=st.session_state['left'] + st.session_state['width']
                highSquare = st.session_state['top'] + st.session_state['height']
                inicTop = w * (st.session_state['left']/600)
                finWidth = w * (endSquare/600)
                inicHigh = h * (st.session_state['top']/400)
                finHigh = h * (highSquare/400)
                cropped = skimg[int(inicHigh):int(finHigh),int(inicTop):int(finWidth)]
                imgG = img_as_ubyte(cropped)
            else:
                imgG = img_as_ubyte(skimg)

            st.image(imgG)
    
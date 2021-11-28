from classes.coordinate import Coordinate
from classes.queue import Queue
import numpy as np
from numpy import median
import pandas as pd
from PIL import Image
import cv2

import streamlit as st
from streamlit_drawable_canvas import st_canvas
from skimage import io, img_as_ubyte
from skimage.color import rgb2hed, hed2rgb

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Valid value true = is usefull
valid = True
#List of valid an invalid points
valids = []
invalids = []
boxes = []

# Specify canvas parameters in application
stroke_width = 1
bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg"])

def click_event(event, x, y, flags, params):
    if (event == cv2.EVENT_LBUTTONDOWN) or (event==cv2.EVENT_RBUTTONDOWN):
        p = Coordinate(x,y)
        if (valid == True):
            valids.append((x,y))
            s_box = x, y
            boxes.append(s_box)
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

def regiongrow(imageSrc,epsilon,start_point : list):

    Q = Queue()
    s = []
    
    print(start_point)
    for i in start_point:
        Q.enque(i)

    image = imageSrc.convert("L")

    st.image(image)
    while not Q.isEmpty():

        t = Q.deque()
        x = t[0]
        y = t[1]
        
        if x < image.size[0]-1 and \
           abs(  image.getpixel( (x + 1 , y) ) - image.getpixel( (x , y) )  ) <= epsilon :
            if not Q.isInside( (x + 1 , y) ) and not (x + 1 , y) in s:
                print( image.getpixel( (x , y) ) )
                print( image.getpixel( (x + 1 , y) ) ) 
                Q.enque( (x + 1 , y) )

                
        if x > 0 and \
           abs(  image.getpixel( (x - 1 , y) ) - image.getpixel( (x , y) )  ) <= epsilon:

            if not Q.isInside( (x - 1 , y) ) and not (x - 1 , y) in s:
                Q.enque( (x - 1 , y) )

                     
        if y < (image.size[1] - 1) and \
           abs(  image.getpixel( (x , y + 1) ) - image.getpixel( (x , y) )  ) <= epsilon:

            if not Q.isInside( (x, y + 1) ) and not (x , y + 1) in s:
                Q.enque( (x , y + 1) )

                    
        if y > 0 and \
           abs(  image.getpixel( (x , y - 1) ) - image.getpixel( (x , y) )  ) <= epsilon:

            if not Q.isInside( (x , y - 1) ) and not (x , y - 1) in s:
                Q.enque( (x , y - 1) )


        if t not in s:
            s.append( t )

            
    image.load()
    putpixel = image.im.putpixel
    
    for i in range ( image.size[0] ):
        for j in range ( image.size[1] ):
            putpixel( (i , j) , 0 )

    for i in s:
        putpixel(i , 150)
        
    return image

def regiongrowMediana(imageSrc,epsilon,start_point : list):
    print('Calculando')
    Q = Queue()
    s = []

    color = []
    
    print(start_point)
    for i in start_point:
        Q.enque(i)

    image = imageSrc.convert("L")

    #Preparado para calcular mediana
    for pixel in start_point:
        x = pixel[0]
        y = pixel[1]
        color.append(image.getpixel( (x , y) ))

    print( color )   
    print( median(color))

    st.image(image)
    while not Q.isEmpty():
        
        #print( median(color))

        t = Q.deque()
        x = t[0]
        y = t[1]
        
        if x < image.size[0]-1 and \
           abs(  image.getpixel( (x + 1 , y) ) - median(color)  ) <= epsilon :
            if not Q.isInside( (x + 1 , y) ) and not (x + 1 , y) in s:
                #print("-------------")
                #print(image.getpixel( (x + 1 , y) ))
                color.append(image.getpixel( (x + 1, y) ))
                Q.enque( (x + 1 , y) )

                
        if x > 0 and \
           abs(  image.getpixel( (x - 1 , y) ) - median(color)  ) <= epsilon:
            if not Q.isInside( (x - 1 , y) ) and not (x - 1 , y) in s:
                #print("-------------")
                #print(image.getpixel( (x -1 , y) ))
                color.append(image.getpixel( (x - 1 , y) ))
                Q.enque( (x - 1 , y) )

                     
        if y < (image.size[1] - 1) and \
           abs(  image.getpixel( (x , y + 1) ) - median(color)  ) <= epsilon:
            if not Q.isInside( (x, y + 1) ) and not (x , y + 1) in s:
                #print("-------------")
                #print(image.getpixel( (x , y + 1)))
                color.append(image.getpixel( (x , y + 1) ))
                Q.enque( (x , y + 1) )

                    
        if y > 0 and \
           abs(  image.getpixel( (x , y - 1) ) - median(color)  ) <= epsilon:
            if not Q.isInside( (x , y - 1) ) and not (x , y - 1) in s:
                #print("-------------")
                #print(image.getpixel( (x , y - 1) ))
                color.append(image.getpixel( (x , y - 1) ))
                Q.enque( (x , y - 1) )


        if t not in s:
            s.append( t )

            
    image.load()
    putpixel = image.im.putpixel
    
    for i in range ( image.size[0] ):
        for j in range ( image.size[1] ):
            putpixel( (i , j) , 0 )

    for i in s:
        putpixel(i , 150)
        
    
    #output=raw_input("enter save fle name : ")
    #image.thumbnail( (image.size[0] , image.size[1]) , Image.ANTIALIAS )
    #image.save(output + ".JPEG" , "JPEG")
    return image

def test_method(imageSrc, start_point : list):
    imageGrey = imageSrc.convert("L")
    st.image(imageGrey)

    newW,newH = imageSrc.size
    print(newW,newH)

    widthGrey,HeightGrey = imageGrey.size
    print(widthGrey,HeightGrey)
    #relationX = newW / widthGrey
    #relationY = newH / HeightGrey
    #print(relationX,relationY)

    for pixel in start_point:
        x = pixel[0]
        y = pixel[1]
        print((x , y))
        print(imageSrc.getpixel( (x , y) ))
        print(imageGrey.getpixel( (x , y) ))

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

    #Subir brillo y contraste
    bright = st.sidebar.slider( "Brillo" , min_value=-127 , max_value=127 , value=64 , step=None , format=None , key=None)
    contrast = st.sidebar.slider( "Contraste" , min_value=-64 , max_value=64 , value=64 , step=None , format=None , key=None)
    
    out = apply_brightness_contrast(img, bright, contrast)
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
            else:
                imgG = img_as_ubyte(skimg)
            
            img  = apply_brightness_contrast(imgG, bright, contrast)
            
            # Separate the stains from the IHC image Numpy
            ihc_hed = rgb2hed(img)
            null = np.zeros_like(ihc_hed[:, :, 0])
            #ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
            #ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
            ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
            #st.image(ihc_h)
            #st.image(ihc_e)
            #st.image(ihc_d)

            #Imagen separada, canal marron en PIL
            pil_image_brown=Image.fromarray((ihc_d * 255).astype(np.uint8))
            st.image(pil_image_brown)

            imgG = img_as_ubyte(pil_image_brown)
            RGB_img = cv2.cvtColor(imgG, cv2.COLOR_BGR2RGB)
            
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            # create switch for ON/OFF functionality
            #cv2.createTrackbar('0 : OFF \n1 : ON', 'img',0,1,switchValid)
            cv2.cvtColor(imgG, cv2.COLOR_BGR2RGB)
            cv2.imshow('img', RGB_img)
            cv2.setMouseCallback('img',click_event)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            print ('Inicio RG')
            #st.image(regiongrow(pil_image_brown,1,valids))
            #test_method(pil_image_brown,valids)
            regionGrowResult = regiongrowMediana(pil_image_brown,35,valids)
            st.image(regionGrowResult)
            print ('fin RG')

            print(regionGrowResult.histogram())
            st.bar_chart(regionGrowResult.histogram())
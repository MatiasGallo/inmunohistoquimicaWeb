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

#Valid value true = is usefull
valid = True
#List of valid an invalid points
valids = []
invalids = []
boxes = []

# Specify canvas parameters in application
stroke_width = 1
bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg", "jpeg"])

if 'bright' not in st.session_state:
    st.session_state['bright'] = 64

if 'contrast' not in st.session_state:
    st.session_state['contrast'] = 64

if 'Epsilon' not in st.session_state:
    st.session_state['Epsilon'] = 25

if 'MinThreshold' not in st.session_state:
    st.session_state['MinThreshold'] = 150

if 'MaxThreshold' not in st.session_state:
    st.session_state['MaxThreshold'] = 256

if 'Area' not in st.session_state:
    st.session_state['Area'] = 625

if 'Circularidad' not in st.session_state:
    st.session_state['Circularidad'] = 0.6

def click_event(event, x, y, flags, params):
    if (event == cv2.EVENT_LBUTTONDOWN) or (event==cv2.EVENT_RBUTTONDOWN):
        valids.append((x,y))
        s_box = x, y
        boxes.append(s_box)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(RGB_img, '.' , ((x-5),y), font,
                    1, (0, 0, 0), 2)
        cv2.imshow('img', RGB_img)
        st.session_state['clicks'] = valids

@st.cache(suppress_st_warning=True)
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

@st.cache(suppress_st_warning=True)
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

@st.cache(suppress_st_warning=True)
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

    #print( color )   
    #print( median(color))
    #st.image(image)

    while not Q.isEmpty(): 
        #print( median(color))

        t = Q.deque()
        x = t[0]
        y = t[1]

        print(x,y)
        
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

@st.cache(suppress_st_warning=True)
def Blob_Detection_Cv2(image):
    #Blob detection
    im = img_as_ubyte(image)

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = st.sidebar.slider( "MinThreshold" , min_value=0 , max_value=1000 , value=st.session_state['MinThreshold'] , step=None , format=None , key=None)
    params.maxThreshold = st.sidebar.slider( "MaxThreshold" , min_value=0 , max_value=2000 , value=st.session_state['MaxThreshold'] , step=None , format=None , key=None)

    params.filterByArea = True
    params.minArea = st.sidebar.slider( "Area" , min_value=0 , max_value=1000 , value=st.session_state['Area'] , step=None , format=None , key=None)

    params.filterByCircularity = True
    params.minCircularity = st.sidebar.slider( "Circularidad" , min_value=0.0 , max_value=1.0 , value=st.session_state['Circularidad'], step=0.1 , format=None , key=None)

    params.filterByColor = True
    params.blobColor = st.sidebar.slider( "Blob Color" , min_value=1 , max_value=255 , value=st.session_state['Blob Color'] , step=1 , format=None , key=None)

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

if bg_image:
    drawing_mode = st.sidebar.selectbox("Drawing tool:", ("line", "rect"))

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

    img = img_as_ubyte(newImage)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    height, width, channels = RGB_img.shape

    if 'left' in st.session_state:
        del st.session_state['left']
    if 'top' in st.session_state:
        del st.session_state['top']
    if 'width' in st.session_state:
        del st.session_state['width']
    if 'height' in st.session_state:
        del st.session_state['height']

    if canvas_result.json_data is not None:
        formas=pd.json_normalize(canvas_result.json_data["objects"])

        for index in range(0, len(formas)):
            if formas['type'][index] == "line":
                left = formas['left'][index]
                top = formas['top'][index]
                x1 = formas['x1'][index]
                x2 = formas['x2'][index]
                y1 = formas['y1'][index]
                y2 = formas['y2'][index]
                inicX = left + x1
                endX2 = left + x2
                inicY = top + y1
                endY  = top + y2

                startingX = newW * (inicX/600)
                finishX = newW * (endX2/600)
                statingY = newH * (inicY/400)
                finishY = newH * (endY/400)
            
                cv2.line(RGB_img, pt1=(int(startingX),int(statingY)), pt2=(int(finishX),int(finishY)), color=(255,255,255), thickness=10)
            elif formas['type'][index] == "rect":
                st.session_state['left'] = formas['left'][index]
                st.session_state['top'] = formas['top'][index]
                st.session_state['width'] = formas['width'][index]
                st.session_state['height'] = formas['height'][index]

        opencvImage2 = cv2.cvtColor(np.array(RGB_img), cv2.COLOR_RGB2BGR)
        im = img_as_ubyte(opencvImage2)
        st.image(im)

        if st.sidebar.button('Recortar'):
            #Borrar seeds si cambia imagen
            if 'clicks' in st.session_state:
                del st.session_state['clicks']
            # plugin='matplotlib'
            skimg = im
            if 'left' in st.session_state:
                endSquare=st.session_state['left'] + st.session_state['width']
                highSquare = st.session_state['top'] + st.session_state['height']
                inicTop = w * (st.session_state['left']/600)
                finWidth = w * (endSquare/600)
                inicHigh = h * (st.session_state['top']/400)
                finHigh = h * (highSquare/400)
                cropped = skimg[int(inicHigh):int(finHigh),int(inicTop):int(finWidth)]
                imgG = img_as_ubyte(cropped)
                st.session_state['img_prepation'] = imgG
            else:
                imgG = img_as_ubyte(skimg)
                st.session_state['img_prepation'] = imgG          

if 'img_prepation' in st.session_state:
    #Subir brillo y contraste
    st.session_state['bright'] = st.sidebar.slider( "Brillo" , min_value=-127 , max_value=127 , value=st.session_state['bright'] , step=None , format=None , key=None)
    st.session_state['contrast'] = st.sidebar.slider( "Contraste" , min_value=-64 , max_value=127 , value=st.session_state['contrast'], step=None , format=None , key=None)

    imgBright  = apply_brightness_contrast(st.session_state['img_prepation'], st.session_state['bright'], st.session_state['contrast'])
    st.session_state['img_brightness_contrast'] = imgBright

if 'img_brightness_contrast' in st.session_state:
    st.text("Brillo")
    st.image(st.session_state['img_brightness_contrast'])

    # Separate the stains from the IHC image Numpy
    ihc_hed = rgb2hed(st.session_state['img_brightness_contrast'])
    null = np.zeros_like(ihc_hed[:, :, 0])
    ihc_h = hed2rgb(np.stack((ihc_hed[:, :, 0], null, null), axis=-1))
    #ihc_e = hed2rgb(np.stack((null, ihc_hed[:, :, 1], null), axis=-1))
    ihc_d = hed2rgb(np.stack((null, null, ihc_hed[:, :, 2]), axis=-1))
    #Azul
    st.image(ihc_h)
    #st.image(ihc_e)
    st.image(ihc_d)

    #st.session_state['ihc_d'] = ihc_d

    #Imagen separada, canal marron en PIL
    pil_image_brown=Image.fromarray((ihc_d * 255).astype(np.uint8))
    st.session_state['pil_image_brown'] = pil_image_brown

if 'pil_image_brown' in st.session_state:
    st.text("Marron")
    st.image(st.session_state['pil_image_brown'])

if 'pil_image_brown' in st.session_state:
    st.sidebar.text("Blob Params")

    im = img_as_ubyte(st.session_state['pil_image_brown'])

    #st.text("test")
    #st.image(im)

    params = cv2.SimpleBlobDetector_Params()

    params.minThreshold = st.sidebar.slider( "MinThreshold" , min_value=0 , max_value=1000 , value=st.session_state['MinThreshold'] , step=None , format=None , key=None)
    params.maxThreshold = st.sidebar.slider( "MaxThreshold" , min_value=0 , max_value=2000 , value=st.session_state['MaxThreshold'] , step=None , format=None , key=None)

    params.filterByArea = True
    params.minArea = st.sidebar.slider( "Area" , min_value=0 , max_value=1500 , value=st.session_state['Area'] , step=None , format=None , key=None)
    params.maxArea = 10000

    params.filterByCircularity = True
    params.minCircularity = st.sidebar.slider( "Circularidad" , min_value=0.0 , max_value=1.0 , value=st.session_state['Circularidad'], step=0.1 , format=None , key=None)
   
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(im)

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
    im_pil = Image.fromarray(im_with_keypoints)
    st.session_state['BlobImage'] = im_pil

    #st.text("test2")
    #st.image(im_pil)
            
    print("keypoints")
    for keyPoint in keypoints:
        x = keyPoint.pt[0]
        y = keyPoint.pt[1]
        s = keyPoint.size
        print(keyPoint)
        print(x)
        print(y)
        print(s)

if 'BlobImage' in st.session_state:
    st.text("Blob")
    st.image(st.session_state['BlobImage'])

if 'pil_image_brown' in st.session_state:
    if st.sidebar.button('Seeds'):
        st.session_state['clicks'] = []

        imgG = img_as_ubyte(st.session_state['pil_image_brown'])
        RGB_img = cv2.cvtColor(imgG, cv2.COLOR_BGR2RGB)
            
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        # create switch for ON/OFF functionality
        #cv2.createTrackbar('0 : OFF \n1 : ON', 'img',0,1,switchValid)
        cv2.cvtColor(imgG, cv2.COLOR_BGR2RGB)
        cv2.imshow('img', RGB_img)
        cv2.setMouseCallback('img',click_event)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if 'clicks' in st.session_state and 'pil_image_brown' in st.session_state:
    st.session_state['Epsilon'] = st.sidebar.slider( "Epsilon" , min_value=1 , max_value=127 , value=st.session_state['Epsilon'] , step=None , format=None , key=None)
    if st.sidebar.button('Region Grow'):

        with st.spinner('Wait for it...'):
            regionGrowResult = regiongrowMediana(st.session_state['pil_image_brown'],st.session_state['Epsilon'],st.session_state['clicks'])

            st.session_state['regionGrowResult'] = regionGrowResult
            #st.image(regionGrowResult)

            #print(regionGrowResult.histogram())
            #st.bar_chart(regionGrowResult.histogram())

if 'regionGrowResult' in st.session_state:
    st.text("RegionGrow")
    st.image(st.session_state['regionGrowResult'])
from classes.queue import Queue
from PIL import Image
import streamlit as st
import cv2
from skimage import img_as_ubyte 
from numpy import median

def regiongrow(imageSrc,epsilon,start_point : list):
    print('Calculando')
    Q = Queue()
    s = []

    color = []
    
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
            #print(image.getpixel( (x + 1 , y) ))
            #print(image.getpixel( (x , y) ))
            #print( image.getpixel( (x + 1 , y) ) - image.getpixel( (x , y) ))
            #print('--------------------------')
            if not Q.isInside( (x + 1 , y) ) and not (x + 1 , y) in s:
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
        
    
    #output=raw_input("enter save fle name : ")
    #image.thumbnail( (image.size[0] , image.size[1]) , Image.ANTIALIAS )
    #image.save(output + ".JPEG" , "JPEG")
    return image

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        print ('Seed: ' + str(x) + ', ' + str(y), img[y,x])
        clicks.append((x,y))
        st.session_state['clicks'] = clicks


bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg"])
pil_image = None
RGB_img = None

clicks = []
if bg_image:
    pil_image = Image.open(bg_image)
    img = img_as_ubyte(pil_image)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(pil_image)
    

if (st.sidebar.button('Seeds') and RGB_img.all()):
    print ('1')
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', RGB_img)
    cv2.setMouseCallback('img',on_mouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print ('2')

epsilon = st.sidebar.slider( "Brillo" , min_value=0 , max_value=1 , value=1 , step=None , format=None , key=None)
if (st.sidebar.button('Region Grow')):
    #st.image(pil_image)
    print ('Inicio RG')
    st.image(regiongrow(pil_image,0,clicks))
    print ('fin RG')

if st.sidebar.button('Ver Clicks [DevTool]'):
    print (st.session_state['clicks'])

from classes.queue import Queue
from PIL import Image
import streamlit as st
import cv2
from skimage import img_as_ubyte 

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


bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg"])

if bg_image:
    pil_image = Image.open(bg_image)
    clicks = []
    img = img_as_ubyte(pil_image)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    # create switch for ON/OFF functionality
    #cv2.createTrackbar('0 : OFF \n1 : ON', 'img',0,1,switchValid)
    #cv2.cvtColor(imgG, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', RGB_img)
    cv2.setMouseCallback('img',on_mouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    st.image(pil_image)
    print ('Inicio RG')
    st.image(regiongrow(pil_image,1.2,clicks))
    print ('fin RG')
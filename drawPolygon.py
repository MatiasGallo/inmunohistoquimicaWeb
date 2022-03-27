from PIL import Image
import streamlit as st
import cv2
from skimage import img_as_ubyte 
import PIL.ImageDraw as ImageDraw

def convertImage(pil_image):
    img = img_as_ubyte(pil_image)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return RGB_img

def drawPoligon(clickName, imgName,img):
    if clickName in st.session_state:
        del st.session_state[clickName]
    params = [clickName]

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('img', cv2.WND_PROP_TOPMOST, 1)
    cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', RGB_img)
    cv2.setMouseCallback('img',on_mouse,params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    draw = ImageDraw.Draw(pil_image)
    draw.polygon((st.session_state[clickName]), fill="#FFFFFF")

    st.session_state[imgName] = pil_image


def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(RGB_img, '.' , ((x-5),y), font,1, (170, 255, 0), 2)
        if clicks:
            last_element = clicks[-1]
            line_thickness = 2
            cv2.line(RGB_img, last_element, (x, y), (170, 255, 0), thickness=line_thickness)


        clicks.append((x,y))
        st.session_state[params[0]] = clicks
        
        cv2.imshow('img', RGB_img)
        

bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg"])

clicks = []
if bg_image:
    pil_image = Image.open(bg_image)
    RGB_img = convertImage(pil_image)
    st.image(pil_image)

if (st.sidebar.button('Marcar 1') and RGB_img.all()):
    if 'Poligon2' in st.session_state:
        del st.session_state['Poligon2']
    
    drawPoligon('clicks','Poligon1', RGB_img)

if 'Poligon1' in st.session_state:
    st.image(st.session_state['Poligon1'])

if (st.sidebar.button('Marcar 2') and RGB_img.all() and 'Poligon1' in st.session_state):
    pil_image = st.session_state['Poligon1'].copy()
    RGB_img = convertImage(pil_image)
    
    drawPoligon('clicks2','Poligon2', RGB_img)

if 'Poligon2' in st.session_state:
    st.image(st.session_state['Poligon2'])

if (st.sidebar.button('Marcar 3') and RGB_img.all()):
    drawPoligon('clicks','Poligon3', RGB_img)

if 'Poligon3' in st.session_state:
    st.image(st.session_state['Poligon3'])
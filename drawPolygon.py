from PIL import Image
import streamlit as st
import cv2
from skimage import img_as_ubyte 
import PIL.ImageDraw as ImageDraw

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(RGB_img, '.' , ((x-5),y), font,1, (170, 255, 0), 2)
        if clicks:
            last_element = clicks[-1]
            line_thickness = 2
            cv2.line(RGB_img, last_element, (x, y), (170, 255, 0), thickness=line_thickness)

        #print ('Click: ' + str(x) + ', ' + str(y), img[y,x])
        clicks.append((x,y))
        st.session_state['clicks'] = clicks
        
        cv2.imshow('img', RGB_img)
        

bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg"])

clicks = []
if bg_image:
    pil_image = Image.open(bg_image)
    img = img_as_ubyte(pil_image)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(pil_image)

if (st.sidebar.button('Marcar') and RGB_img.all()):
    if 'clicks' in st.session_state:
        del st.session_state['clicks']
    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('img', cv2.WND_PROP_TOPMOST, 1)
    cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', RGB_img)
    cv2.setMouseCallback('img',on_mouse)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if st.sidebar.button('Poligono'):
    print(st.session_state['clicks'])

    draw = ImageDraw.Draw(pil_image)
    draw.polygon((st.session_state['clicks']), fill="#FFFFFF")

    st.image(pil_image)
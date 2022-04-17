from PIL import Image
import streamlit as st
import cv2
from skimage import img_as_ubyte 
import PIL.ImageDraw as ImageDraw
from PIL import ImageColor
import numpy as np

def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return [h, s, v]

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

def pick_Color(clickName):
    if clickName in st.session_state:
        del st.session_state[clickName]
    params = [clickName]

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('img', cv2.WND_PROP_TOPMOST, 1)
    cv2.cvtColor(RGB_img, cv2.COLOR_BGR2RGB)
    cv2.imshow('img', RGB_img)
    cv2.setMouseCallback('img',on_mouse_color,params)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def on_mouse(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(RGB_img, '.' , ((x-5),y), font,1, (170, 255, 0), 2)
        if clicks:
            last_element = clicks[-1]
            line_thickness = 2
            cv2.line(RGB_img, last_element, (x, y), (170, 255, 0), thickness=line_thickness)

        #value = RGB_img[y,x]
        #print(value)

        clicks.append((x,y))
        st.session_state[params[0]] = clicks
        
        cv2.imshow('img', RGB_img)

def on_mouse_color(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(RGB_img, '.' , ((x),y), font,1, (170, 255, 0), 2)

        value = img[y,x]
        #print(y,x)
        #print(value)

        st.session_state[params[0]] = value
        
        cv2.imshow('img', RGB_img)

def checkColor(img, colorMin, colorMax):
    img = img_as_ubyte(img)
    frame = img

    # use rgb color picker to set these based on color range you want
    # (order is BGR not RGB!)
    lower_brown = np.array(colorMin) # a dark brown
    upper_brown = np.array(colorMax) # BGR of your brown

    #print("RGB")
    #print(lower_brown)
    #print(upper_brown)
    '''
    print("HSV")
    hsv_min = rgb_to_hsv(lower_brown[0],lower_brown[1],lower_brown[2])
    hsv_max = rgb_to_hsv(upper_brown[0],upper_brown[1],upper_brown[2])

    np_lower_brown = np.array(hsv_min) # a dark brown
    np_upper_brown = np.array(hsv_max) # BGR of your brown

    print(np_lower_brown)
    print(np_upper_brown)
    '''
    w,h,c = frame.shape
    #print("Frame")
    #print(w)
    #print(h)
    mask = cv2.inRange(frame, lower_brown, upper_brown)
    num_brown = cv2.countNonZero(mask)
    perc_brown = num_brown/float(w*h)*100
    st.text("Total pixels")
    st.text(w*h)
    st.text("Values")
    st.text(num_brown)
    st.text(perc_brown)

bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg"])

clicks = []
if bg_image:
    pil_image = Image.open(bg_image)
    img = img_as_ubyte(pil_image)
    RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(pil_image)

if (st.sidebar.button('Agregar Poligono') and RGB_img is not None):
    if 'Poligon1' in st.session_state:
        pil_image = st.session_state['Poligon1'].copy()
    RGB_img = convertImage(pil_image)
    
    drawPoligon('clicks1','Poligon1', RGB_img)

if 'Poligon1' in st.session_state:
    st.image(st.session_state['Poligon1'])

#colorMin = st.sidebar.color_picker('Pick A Color min brown', '#6E6E6E')
if (st.sidebar.button('Color Minimo (claro)') and RGB_img is not None):
    pick_Color("minRGB")

if 'minRGB' in st.session_state:
    st.sidebar.image(Image.new('RGB', (50, 50), (st.session_state['minRGB'][0],st.session_state['minRGB'][1],st.session_state['minRGB'][2])))

#colorMax = st.sidebar.color_picker('Pick A Color max brown', '#8C968C')
if (st.sidebar.button('Color Maximo (oscuro)') and RGB_img is not None):
    pick_Color("maxRGB")

if 'maxRGB' in st.session_state:
    st.sidebar.image(Image.new('RGB', (50, 50), (st.session_state['maxRGB'][0],st.session_state['maxRGB'][1],st.session_state['maxRGB'][2])))

if (st.sidebar.button('Calcular') and 'Poligon1' in st.session_state and 'maxRGB' in st.session_state and 'minRGB' in st.session_state):
    checkColor(st.session_state['Poligon1'], st.session_state['maxRGB'], st.session_state['minRGB'])

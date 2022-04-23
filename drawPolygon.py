from PIL import Image
import streamlit as st
import cv2
from skimage import img_as_ubyte 
import PIL.ImageDraw as ImageDraw
import numpy as np
import pandas as pd

if ('minRGB' not in st.session_state):
    st.session_state['minRGB']=np.array([255, 255, 255], dtype=np.uint8)

if ('maxRGB' not in st.session_state):
    st.session_state['maxRGB']=np.array([0, 0, 0], dtype=np.uint8)

if ('dataFrame_name' not in st.session_state):
    st.session_state['dataFrame_name'] = {}
    st.session_state['dataFrame_total'] = {}
    st.session_state['dataFrame_total_encontrados'] = {}
    st.session_state['dataFrame_perc_encontrados'] = {}

def cleanState():
    if 'imgPoligono' in st.session_state:
        del st.session_state['imgPoligono']
    if 'totalPixeles' in st.session_state:
        del st.session_state['totalPixeles']
    if 'cantDetectada' in st.session_state:
        del st.session_state['cantDetectada']
    if 'percDetectado' in st.session_state:
        del st.session_state['percDetectado']
    if 'ImagenResultado' in st.session_state:
        del st.session_state['ImagenResultado']
    if 'clicksList' in st.session_state:
        del st.session_state['clicksList']
    if 'RGB_img' in st.session_state:
        del st.session_state['RGB_img']

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

    if clickName in st.session_state:
        clicks = list(st.session_state[clickName])
        print(len(clicks))
        if len(clicks) > 1:
            draw = ImageDraw.Draw(pil_image)
            draw.polygon((st.session_state[clickName]), fill="#FFFFFF")
            st.session_state[imgName] = pil_image
        else:
            st.warning("No se agregaron puntos suficientes / la imagen no se afecto")
    else:
        st.warning("No se detecto ningun punto / la imagen no se afecto")

def pick_Color(clickName, RGB_img):
    if clickName in st.session_state:
        del st.session_state[clickName]
    params = [clickName]

    cv2.namedWindow('img', cv2.WINDOW_NORMAL)
    cv2.setWindowProperty('img', cv2.WND_PROP_TOPMOST, 1)
    cv2.cvtColor(st.session_state['RGB_img'], cv2.COLOR_BGR2RGB)
    cv2.imshow('img', st.session_state['RGB_img'])
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
        cv2.putText(st.session_state['RGB_img'], '.' , ((x),y), font,1, (170, 255, 0), 2)

        value = img[y,x]
        #print(y,x)
        #print(value)

        st.session_state[params[0]] = value
        
        cv2.imshow('img', st.session_state['RGB_img'])

def cv2_hsvChange(hsvSrc):
    #(H/2, (S/100) * 255, (V/100) * 255) 
    newH = hsvSrc[0] / 2
    newS = (hsvSrc[1]/100) * 255
    newV = (hsvSrc[2]/100) * 255
    return (newH,newS,newV)

def minToMax_HSV(hsvMinSrc,hsvMaxSrc):
    hsvMin = list(hsvMinSrc)
    hsvMax = list(hsvMaxSrc)

    if hsvMin[0] > hsvMax[0]:
        temp = hsvMin[0]
        hsvMin[0] = hsvMax[0]
        hsvMax[0] = temp
    if hsvMin[1] > hsvMax[1]:
        temp = hsvMin[1]
        hsvMin[1] = hsvMax[1]
        hsvMax[1] = temp
    if hsvMin[2] > hsvMax[2]:
        temp = hsvMin[2]
        hsvMin[2] = hsvMax[2]
        hsvMax[2] = temp

    hsvMinSrc = tuple(hsvMin)
    hsvMaxSrc = tuple(hsvMax)

    return (hsvMinSrc, hsvMaxSrc)

def checkColorHSV(img, colorMin, colorMax):
    img = img_as_ubyte(img)
    hsvFrame = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    #st.image(hsvFrame)

    lower_brown = np.array(colorMin)
    upper_brown = np.array(colorMax)

    print("HSV")
    hsv_min = rgb_to_hsv(lower_brown[0],lower_brown[1],lower_brown[2])
    hsv_max = rgb_to_hsv(upper_brown[0],upper_brown[1],upper_brown[2])

    np_lower_brown = np.array(hsv_min)
    np_upper_brown = np.array(hsv_max)

    print(np_lower_brown)
    print(np_upper_brown)

    #print("Conversion HSV")
    new_hsv_min = cv2_hsvChange(hsv_min)
    new_hsv_max = cv2_hsvChange(hsv_max)

    #print("minToMax_HSV")
    new_hsv_min,new_hsv_max = minToMax_HSV(new_hsv_min,new_hsv_max)

    np_conv_lower_brown = np.array(new_hsv_min)
    np_conv_upper_brown = np.array(new_hsv_max)

    print(np_conv_lower_brown)
    print(np_conv_upper_brown)

    w,h,c = img.shape
    mask = cv2.inRange(hsvFrame, np_conv_lower_brown, np_conv_upper_brown)
    num_brown = cv2.countNonZero(mask)
    perc_brown = num_brown/float(w*h)*100
    st.text("Total pixels")
    st.text(w*h)
    st.text("Values")
    st.text(num_brown)
    st.text(perc_brown)

def checkColor(img, colorMin, colorMax):
    img = img_as_ubyte(img)
    frame = img

    lower_brown = np.array(colorMin)
    upper_brown = np.array(colorMax)

    #print("RGB")
    #print(colorMin)
    #print(colorMax)

    w,h,c = frame.shape
    #print("Frame")
    #print(w)
    #print(h)
    mask = cv2.inRange(frame, lower_brown, upper_brown)
    #mask = cv2.inRange(hsvFrame, np_conv_lower_brown, np_conv_upper_brown)
    num_brown = cv2.countNonZero(mask)
    perc_brown = num_brown/float(w*h)*100

    result = cv2.bitwise_and(frame, frame, mask=mask)

    st.session_state['totalPixeles'] = w*h
    st.session_state['cantDetectada'] = num_brown
    st.session_state['percDetectado'] = perc_brown
    st.session_state['ImagenResultado'] = result

def convert_df():
    d = {
     'Nombre': st.session_state['dataFrame_name'],
     'Total Pixeles': st.session_state['dataFrame_total'],
     'Pixeles Detectados': st.session_state['dataFrame_total_encontrados'],
     '% Pixeles Detectados': st.session_state['dataFrame_perc_encontrados']
    }

    df = pd.DataFrame(data=d)
    return df.to_csv(index=False).encode('utf-8')

def add_to_List(sesion_name, value):
    list_State = st.session_state[sesion_name]
    result_list = list(list_State)
    result_list.append(value)
    st.session_state[sesion_name] = result_list

bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg"])

clicks = []
if bg_image:
    pil_image = Image.open(bg_image)
    img = img_as_ubyte(pil_image)
    st.session_state['RGB_img'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(pil_image)
    if 'imgPoligono' not in st.session_state:
        st.session_state['imgPoligono'] = pil_image
else:
    cleanState()

if (st.sidebar.button('Agregar Poligono')):
    if 'imgPoligono' in st.session_state:
        pil_image = st.session_state['imgPoligono'].copy()
        RGB_img = convertImage(pil_image)
        drawPoligon('clicksList','imgPoligono', RGB_img)

if 'imgPoligono' in st.session_state:
    st.image(st.session_state['imgPoligono'])

if (st.sidebar.button('Color Minimo (claro)') and 'RGB_img' in st.session_state):
    pick_Color("minRGB", st.session_state['RGB_img'])

if 'minRGB' in st.session_state:
    st.sidebar.image(Image.new('RGB', (50, 50), (st.session_state['minRGB'][0],st.session_state['minRGB'][1],st.session_state['minRGB'][2])))

if (st.sidebar.button('Color Maximo (oscuro)') and 'RGB_img' in st.session_state):
    pick_Color("maxRGB", st.session_state['RGB_img'])

if 'maxRGB' in st.session_state:
    st.sidebar.image(Image.new('RGB', (50, 50), (st.session_state['maxRGB'][0],st.session_state['maxRGB'][1],st.session_state['maxRGB'][2])))

if (st.sidebar.button('Calcular') and 'imgPoligono' in st.session_state and 'minRGB' in st.session_state):
    checkColor(st.session_state['imgPoligono'], st.session_state['maxRGB'], st.session_state['minRGB'])

if 'ImagenResultado' in st.session_state:
    st.text("Total pixels")
    st.text(st.session_state['totalPixeles'])
    st.text("Values")
    st.text(st.session_state['cantDetectada'])
    st.text(st.session_state['percDetectado'])

    st.image(st.session_state['ImagenResultado'])

if st.sidebar.button('Iniciar Reporte'):
    st.session_state['dataFrame_name'] = {}
    st.session_state['dataFrame_total'] = {}
    st.session_state['dataFrame_total_encontrados'] = {}
    st.session_state['dataFrame_perc_encontrados'] = {}
    st.sidebar.success('Reporte Iniciado')

if st.sidebar.button('Agregar Dato'):
    if 'ImagenResultado' in st.session_state:
        add_to_List('dataFrame_name', bg_image.name)
        add_to_List('dataFrame_total', st.session_state['totalPixeles'])
        add_to_List('dataFrame_total_encontrados', st.session_state['cantDetectada'])
        add_to_List('dataFrame_perc_encontrados', st.session_state['percDetectado'])
        st.sidebar.success('Dato añadido')

st.sidebar.download_button(
   "Descargar reporte",
   convert_df(),
   "Reporte.csv",
   "text/csv",
   key='download-csv'
)
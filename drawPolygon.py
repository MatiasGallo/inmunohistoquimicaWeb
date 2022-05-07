from PIL import Image
import streamlit as st
import cv2
from skimage import img_as_ubyte 
from streamlit_drawable_canvas import st_canvas
import PIL.ImageDraw as ImageDraw
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

if ('total_polygons_area' not in st.session_state):
    st.session_state['total_polygons_area']=0

if ('RGB_type' not in st.session_state):
    st.session_state['RGB_type']=1

if ('minRGB' not in st.session_state):
    st.session_state['minRGB']=np.array([255, 255, 255], dtype=np.uint8)
    st.session_state['minPickRGB']=np.array([255, 255, 255], dtype=np.uint8)

if ('maxRGB' not in st.session_state):
    st.session_state['maxRGB']=np.array([0, 0, 0], dtype=np.uint8)
    st.session_state['maxPickRGB']=np.array([0, 0, 0], dtype=np.uint8)

if ('dataFrame_name' not in st.session_state):
    st.session_state['dataFrame_name'] = {}
    st.session_state['dataFrame_total'] = {}
    st.session_state['dataFrame_total_encontrados'] = {}
    st.session_state['dataFrame_perc_encontrados'] = {}
    st.session_state['color_min'] = {}
    st.session_state['color_max'] = {}

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
    if 'total_polygons_area' in st.session_state:
        del st.session_state['total_polygons_area']

def checkColor(img, colorMin, colorMax):
    img = img_as_ubyte(img)
    frame = img

    lower_brown = np.array(colorMin)
    upper_brown = np.array(colorMax)

    w,h,c = frame.shape

    size = w * h - st.session_state['total_polygons_area'] 

    mask = cv2.inRange(frame, lower_brown, upper_brown)
    num_brown = cv2.countNonZero(mask)
    perc_brown = num_brown/float(size)*100

    result = cv2.bitwise_and(frame, frame, mask=mask)

    st.session_state['totalPixeles'] = size
    st.session_state['cantDetectada'] = num_brown
    st.session_state['percDetectado'] = round(perc_brown, 2)
    st.session_state['ImagenResultado'] = result

def convert_df():
    d = {
     'Nombre': st.session_state['dataFrame_name'],
     'Total Pixeles': st.session_state['dataFrame_total'],
     'Pixeles Detectados': st.session_state['dataFrame_total_encontrados'],
     '% Pixeles Detectados': st.session_state['dataFrame_perc_encontrados'],
     'Color Minimo': st.session_state['color_min'],
     'Color Maximo': st.session_state['color_max']
    }

    df = pd.DataFrame(data=d)
    return df.to_csv(index=False).encode('utf-8')

def add_to_List(sesion_name, value):
    list_State = st.session_state[sesion_name]
    result_list = list(list_State)
    result_list.append(value)
    st.session_state[sesion_name] = result_list

bg_image = st.sidebar.file_uploader("Image:", type=["png", "jpg"])

if bg_image:
    st.markdown("<h2 style='text-align: center; color: grey;'>Imagen Elegida</h2>", unsafe_allow_html=True)

    pil_image = Image.open(bg_image)
    img = img_as_ubyte(pil_image)

    if 'imgPoligono' not in st.session_state:
        st.session_state['imgPoligono'] = pil_image

    newW,newH = pil_image.size
    st.session_state['widthRelation'] = (newW/600)
    st.session_state['heightRelation'] = (newH/400)
    drawing_mode = "polygon"
    stroke_width = 1
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",
        stroke_width=stroke_width,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit='true',
        drawing_mode=drawing_mode,
        #Default hight 400
        #Default width 600
        key="canvas",
    )
else:
    cleanState()

if (st.sidebar.button('Agregar Poligonos') and 'imgPoligono' in st.session_state):
    if canvas_result.json_data is not None:
        formas=pd.json_normalize(canvas_result.json_data["objects"])

        polygons = []
        for index in range(0, len(formas)):
            path = formas['path'][index]
            polygon = []
            for index in range(0, len(path)-1):
                polygon.append((int(path[index][1] * st.session_state['widthRelation']),int(path[index][2] * st.session_state['heightRelation']))) 
            if len(polygon) > 1:
                polygons.append(polygon)

        area_poligonos = 0
        for polygon in polygons:
            if len(polygon) > 0:
                pil_image = st.session_state['imgPoligono']
                draw = ImageDraw.Draw(pil_image)
                draw.polygon(polygon, fill="#FFFFFF")
                st.session_state['imgPoligono'] = pil_image

                pgon = Polygon(polygon)
                area_poligonos = area_poligonos + pgon.area
        
        st.session_state['total_polygons_area'] = area_poligonos

if 'imgPoligono' in st.session_state:
    st.markdown("<h2 style='text-align: center; color: grey;'>Imagen Recortada</h2>", unsafe_allow_html=True)
    st.image(st.session_state['imgPoligono'])

if bg_image:
    st.markdown("<h2 style='text-align: center; color: grey;'>Elegir Colores</h2>", unsafe_allow_html=True)

    pil_image = Image.open(bg_image)
    pix = pil_image.load()
    img = img_as_ubyte(pil_image)

    canvas_color = st_canvas(
        fill_color="rgba(255, 165, 0, 0)",  # Fixed fill color with some opacity
        stroke_color="rgba(170, 255, 0, 1)",
        background_image=Image.open(bg_image) if bg_image else None,
        stroke_width = 1,
        update_streamlit='true',
        drawing_mode="circle",
        #Default hight 400
        #Default width 600
        key="canvas_color",
    )

    if canvas_color.json_data is not None:
        formas=pd.json_normalize(canvas_color.json_data["objects"])
        for index in range(0, len(formas)):
            left = int(formas['left'][index]*st.session_state['widthRelation'])
            top = int(formas['top'][index]*st.session_state['heightRelation'])
            rgb = pix[left,top]
            if st.session_state['RGB_type'] == 1:
                st.session_state['minRGB'] = np.array(rgb, dtype=np.uint8)
            else:
                st.session_state['maxRGB'] = np.array(rgb, dtype=np.uint8)

if (st.sidebar.button('Color Minimo (claro)')):
    st.session_state['RGB_type'] = 1

if 'minRGB' in st.session_state:
    st.sidebar.image(Image.new('RGB', (50, 50), (st.session_state['minRGB'][0],st.session_state['minRGB'][1],st.session_state['minRGB'][2])))

if (st.sidebar.button('Color Maximo (oscuro)')):
    st.session_state['RGB_type'] = 0

if 'maxRGB' in st.session_state:
    st.sidebar.image(Image.new('RGB', (50, 50), (st.session_state['maxRGB'][0],st.session_state['maxRGB'][1],st.session_state['maxRGB'][2])))

if (st.sidebar.button('Calcular') and 'imgPoligono' in st.session_state and 'minRGB' in st.session_state):
    checkColor(st.session_state['imgPoligono'], st.session_state['maxRGB'], st.session_state['minRGB'])
    st.session_state['minPickRGB'] = st.session_state['minRGB']
    st.session_state['maxPickRGB'] = st.session_state['maxRGB']

if 'ImagenResultado' in st.session_state:
    st.markdown("<h2 style='text-align: center; color: grey;'>Imagen Resultado</h2>", unsafe_allow_html=True)
    st.image(st.session_state['ImagenResultado'])
    st.markdown("<h2 style='text-align: center; color: grey;'>Resultados</h2>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: grey;'>" + "Total pixels: " + str(st.session_state['totalPixeles']) +"</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: grey;'>" + "Pixels detectados: " + str(st.session_state['cantDetectada']) +"</h5>", unsafe_allow_html=True)
    st.markdown("<h5 style='text-align: center; color: grey;'>" + "% detectados: " + str(st.session_state['percDetectado']) +"</h5>", unsafe_allow_html=True)

st.sidebar.text("---------------------------------")

if st.sidebar.button('Iniciar Reporte'):
    st.session_state['dataFrame_name'] = {}
    st.session_state['dataFrame_total'] = {}
    st.session_state['dataFrame_total_encontrados'] = {}
    st.session_state['dataFrame_perc_encontrados'] = {}
    st.session_state['color_min'] = {}
    st.session_state['color_max'] = {}
    st.sidebar.success('Reporte Iniciado')

if st.sidebar.button('Agregar Dato'):
    if 'ImagenResultado' in st.session_state:
        add_to_List('dataFrame_name', bg_image.name)
        add_to_List('dataFrame_total', st.session_state['totalPixeles'])
        add_to_List('dataFrame_total_encontrados', st.session_state['cantDetectada'])
        add_to_List('dataFrame_perc_encontrados', st.session_state['percDetectado'])
        add_to_List('color_min', st.session_state['minPickRGB'])
        add_to_List('color_max', st.session_state['maxPickRGB'])
        st.sidebar.success('Dato añadido')

st.sidebar.download_button(
   "Descargar reporte",
   convert_df(),
   "Reporte.csv",
   "text/csv",
   key='download-csv'
)
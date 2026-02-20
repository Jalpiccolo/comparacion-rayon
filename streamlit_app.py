import streamlit as st
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage import color
from PIL import Image
import os

# --- Configuración de la página ---
st.set_page_config(page_title="Compara Rayón - Piccolo", page_icon="🧵", layout="wide")

# --- Estilos CSS ---
st.markdown("""
    <style>
    /* Estilo general */
    .stApp {
        background-color: #FAFAFA;
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    
    /* Títulos limpios */
    h1, h2, h3 {
        color: #333333;
    }
    
    /* Botones Piccolo (Rosa/Pastel) */
    .stButton>button, .stLinkButton>a {
        background-color: #F8C8DC !important; /* Rosa pastel */
        color: #555555 !important;
        border: none !important;
        font-weight: bold !important;
        border-radius: 8px !important;
        transition: 0.3s !important;
    }
    .stButton>button:hover, .stLinkButton>a:hover {
        background-color: #F4A4C8 !important; /* Rosa más intenso */
        color: white !important;
        text-decoration: none !important;
    }
    
    /* Cajas de color */
    .color-box {
        width: 100%;
        height: 60px;
        border-radius: 8px;
        margin-bottom: 5px;
        border: 1px solid #ddd;
    }
    
    /* Tarjetas de resultados */
    .result-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        text-align: center;
        height: 100%;
        margin-bottom: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# --- Funciones Auxiliares ---

@st.cache_data(show_spinner=False)
def cargar_datos(ruta_csv="base_datos_rayon.csv"):
    try:
        df = pd.read_csv(ruta_csv)
        # Parsear la columna Color_RGB (asumiendo formato "(R, G, B)" o "[R, G, B]")
        def parse_rgb(rgb_str):
            try:
                rgb_str = str(rgb_str).strip('()[]{}')
                parts = rgb_str.split(',')
                return np.array([int(p.strip()) for p in parts])
            except:
                return np.array([0, 0, 0])
                
        df['RGB_Array'] = df['Color_RGB'].apply(parse_rgb)
        
        # Convertir colores del inventario a LAB para optimizar
        inventario_rgb = np.vstack(df['RGB_Array'].values).reshape(1, -1, 3).astype(np.uint8)
        inventario_lab = color.rgb2lab(inventario_rgb)[0]
        return df, inventario_lab
    except FileNotFoundError:
        st.error(f"❌ No se encontró la base de datos: `{ruta_csv}`. Asegúrate de que el archivo exista en la misma ruta.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Error al cargar los datos: {e}")
        st.stop()

def redimensionar_imagen(imagen, max_height=400):
    # Evita problemas de memoria redimensionando si es muy grande
    h, w = imagen.shape[:2]
    if h > max_height:
        ratio = max_height / h
        nueva_w = int(w * ratio)
        return cv2.resize(imagen, (nueva_w, max_height))
    return imagen

def extraer_colores(imagen_np, k=10):
    # Reformatear a lista de píxeles
    pixels = imagen_np.reshape((-1, 3))
    
    # Submuestreo si hay muchos píxeles para ahorrar memoria y tiempo en Streamlit Cloud
    if len(pixels) > 50000:
        indices = np.random.choice(len(pixels), 50000, replace=False)
        pixels = pixels[indices]
        
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    colores_rgb = kmeans.cluster_centers_.astype(int)
    
    # Calcular proporción de cada color para ordenamiento
    labels = kmeans.labels_
    counts = np.bincount(labels)
    indices_ordenados = np.argsort(counts)[::-1]
    
    return colores_rgb[indices_ordenados]

def rgb_a_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def encontrar_hilos_cercanos(color_objetivo_rgb, df_inventario, inventario_lab, n=2):
    # Convertir color objetivo a LAB
    objetivo_rgb_arr = np.array([[color_objetivo_rgb]], dtype=np.uint8)
    objetivo_lab = color.rgb2lab(objetivo_rgb_arr)[0][0]
    
    # Calcular Distancia Delta E
    deltas = color.deltaE_cie76(objetivo_lab, inventario_lab)
    
    # Obtener índices de los n más cercanos
    indices_cercanos = np.argsort(deltas)[:n]
    
    resultados = []
    for idx in indices_cercanos:
        row = df_inventario.iloc[idx]
        resultados.append({
            'Nombre_Hilo': row['Nombre_Hilo'],
            'Color_Hex': row['Color_Hex'],
            'Ruta_Local': row['Ruta_Local'] if 'Ruta_Local' in row else '',
            'Distancia': float(deltas[idx])
        })
    return resultados

# --- Layout Principal ---

with st.sidebar:
    if os.path.exists("logo-piccolo.png"):
        st.image("logo-piccolo.png")
    st.markdown("#### 🧵 Tu asistente de costura")
    
    # Caja de instrucciones emulando el diseño de la imagen
    st.markdown("""
    <div style="background-color: #EBF4FC; padding: 15px; border-radius: 10px; margin-top: 10px; margin-bottom: 20px;">
        <h4 style="color: #0E497B; margin-top: 0; font-size: 15px;">¿Cómo funciona?</h4>
        <ol style="color: #0E497B; font-size: 14px; padding-left: 20px; margin-bottom: 0;">
            <li style="margin-bottom: 8px;">Sube una foto de tu proyecto.</li>
            <li style="margin-bottom: 8px;">Detectamos los colores clave.</li>
            <li>Te sugerimos los mejores hilos de nuestro inventario.</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>"*5, unsafe_allow_html=True)
    st.markdown("<hr style='margin: 0;'>", unsafe_allow_html=True)
    st.markdown("<p style='font-size: 13px; color: #333; padding-top: 10px;'>Desarrollado para Piccolo Ind. SAS</p>", unsafe_allow_html=True)

st.title("Descubre tus Hilos Rayon Ideales")
st.markdown("Sube la foto de tu proyecto y deja que nuestra IA encuentre la combinación perfecta de nuestra colección.")

# Carga de datos base
df_inventario, inventario_lab = cargar_datos()

# Subida de archivo
archivo_subido = st.file_uploader("Sube la foto de tu proyecto (JPG, PNG)", type=["jpg", "jpeg", "png"])

if archivo_subido is None:
    st.info("👆 Sube una imagen para comenzar.")
    
    st.markdown("<br><hr style='margin-bottom:10px;'>", unsafe_allow_html=True)
    with st.expander("🔧 Ver código para insertar en WordPress"):
        st.code('<iframe src="URL_DE_TU_APP_STREAMLIT" width="100%" height="800" style="border:none;"></iframe>', language='html')

if archivo_subido is not None:
    # Límite de 5MB para Cloud
    MAX_FILE_SIZE = 5 * 1024 * 1024
    if len(archivo_subido.getvalue()) > MAX_FILE_SIZE:
        st.error("⚠️ La imagen es demasiado pesada. Por favor, sube una imagen menor a 5MB.")
        st.stop()
        
    try:
        imagen = Image.open(archivo_subido).convert('RGB')
        imagen_np = np.array(imagen)
        
        st.markdown("---")
        st.subheader("🖼️ Tu Proyecto")
        
        col_img1, col_img2, col_img3 = st.columns([1, 2, 1])
        with col_img2:
            st.image(imagen, caption="Imagen Original", use_container_width=True)
            
        st.markdown("---")
        st.subheader("✨ Sugerencias de Hilos")
        
        # UI de Progreso
        with st.status("Procesando imagen...", expanded=True) as status:
            st.write("Analizando la paleta de tu diseño... 🎨")
            img_procesar = redimensionar_imagen(imagen_np, max_height=300)
            colores_detectados = extraer_colores(img_procesar, k=10)
            
            st.write("Buscando hilos de Rayón en el inventario... 🧵")
            sugerencias_por_color = []
            for col_rgb in colores_detectados:
                sugerencias = encontrar_hilos_cercanos(col_rgb, df_inventario, inventario_lab, n=2)
                sugerencias_por_color.append((col_rgb, sugerencias))
                
            st.write("Preparando tus sugerencias... ✨")
            status.update(label="¡Búsqueda completada!", state="complete", expanded=False)
            
        # Cuadrícula Visual
        for idx, (color_detectado, sugerencias) in enumerate(sugerencias_por_color):
            hex_detectado = rgb_a_hex(color_detectado)
            
            st.markdown(f"#### 🎨 Color Detectado #{idx + 1}")
            cols = st.columns([1, 2, 2])
            
            with cols[0]:
                st.markdown(f"""
                <div class="result-card">
                    <p style="margin-bottom:5px; font-weight:bold; color:#555; font-size: 14px;">Original</p>
                    <div class="color-box" style="background-color: {hex_detectado};"></div>
                    <p style="margin-top:5px; font-size:12px; color:#777;">{hex_detectado}</p>
                </div>
                """, unsafe_allow_html=True)
                
            for i, sugerencia in enumerate(sugerencias):
                with cols[i + 1]:
                    img_html = f'<img src="{sugerencia["Ruta_Local"]}" style="width:100%; border-radius:8px; margin-bottom:10px;">' if sugerencia['Ruta_Local'] else ''
                    st.markdown(f"""
                    <div class="result-card">
                        <p style="margin-bottom:5px; font-weight:bold; color:#555; font-size: 14px;">Opción {i + 1}: {sugerencia['Nombre_Hilo']}</p>
                        {img_html}
                        <div class="color-box" height="30px" style="background-color: {sugerencia['Color_Hex']};"></div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.link_button("🛒 Ver en Tienda", "https://piccolo.com.co/categoria-producto/bordado/hilos/?product-page=2", use_container_width=True)
            
            st.markdown("<hr style='margin: 10px 0; border: 0; border-top: 1px solid #eee;'>", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Ocurrió un error al procesar la imagen: {e}. Por favor intenta con otra.")

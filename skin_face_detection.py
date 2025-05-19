import numpy as np
import cv2 
from scipy.stats import multivariate_normal

# Medias y covarianza obtenidas de tareas anteriores

# Bandera para usar formato RGB o HSV
formato_hsv = True

if formato_hsv:
       # Datos en HSV, eliminando el canal V
       skin_mean = np.array([10.12600178, 98.29919858])
       skin_cov = np.array([[  5.88897499,  -3.92029384],
              [ -3.92029384, 396.50657795]])

       fondo_mean = np.array([44.84623407, 51.63395133])
       fondo_cov = np.array([[1732.49356271, -370.70399196],
              [-370.70399196, 1929.23776246]])
else:
       # Datos en RGB
       skin_mean = np.array([109.04378154, 130.19204512, 172.47343425])
       skin_cov = np.array([[1557.55025745, 1630.09297116, 1814.77250065],
              [1630.09297116, 1731.13069521, 1941.66919639],
              [1814.77250065, 1941.66919639, 2250.5592615 ]])

       fondo_mean = np.array([ 95.55063731, 106.95504056, 110.26025492])
       fondo_cov = np.array([[4486.90316045, 4649.32282388, 4487.6689383 ],
              [4649.32282388, 5074.34657037, 4956.60725216],
              [4487.6689383 , 4956.60725216, 5081.15951645]])




# Se crea el modelo para el analisis de la imagen
piel_pdf = multivariate_normal(skin_mean, skin_cov).pdf
fondo_pdf = multivariate_normal(fondo_mean, fondo_cov).pdf

# Probabilidades obtenidas de tareas anteriores
priori_piel = 0.5
priori_fondo = 1 - priori_piel

def get_face_roi(img):
    '''Funcion que recibe una imagen en formato array de numpy,
    detecta la piel en la imagen y devuelve coordenadas indicando el
    contorno y la posicion media de la piel detectada'''

    # Calculamos la probabilidad de ser fondo o piel para cada pixel
    # en la imagen
    img_piel_prob = piel_pdf(img) * priori_piel
    img_fondo_prob = fondo_pdf(img) * priori_fondo
    
    # Verificamos que probabilidad es mayor para cada pixel
    skin_pixels = np.asarray(img_piel_prob > img_fondo_prob).nonzero()
    

    # Obtenemos las coordenadas minimas y maximas para los pixeles 
    # detectados como piel
    try:
       min_row, max_row = min(skin_pixels[0]), max(skin_pixels[0])
       min_col, max_col = min(skin_pixels[1]), max(skin_pixels[1])
    except ValueError:
           min_row, max_row, min_col, max_col = (0,0,0,0)
    
    
    
    # Se obtiene el centroide de estos pixeles
    mid_row = (min_row + max_row) // 2
    mid_col = (min_col + max_col) // 2
    
    # Empaquetamos los datos para regresarlos
    top_left = (min_col, min_row) # Top left
    bottom_left = (min_col, max_row) # Bottom left
    top_right = (max_col, min_row) # Top right
    bottom_right = (max_col, max_row) # Bottom right
    
    mid_point = (mid_col, mid_row)
    
    return top_left, top_right, bottom_left, bottom_right, mid_point


# Ejemplo de uso del modelo para detección de piel en una imagen
# estática
if __name__ == '__main__':
      # Se lee la imagen y se transforma al formato HSV, y se toman solamente los canales HS 
      img_bgr = cv2.imread('cara_5.JPEG')
      img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
      img_hs = img_hsv[:,:,:2]
      
      img_piel_prob = piel_pdf(img_hs) * 0.40 # Priori de piel en la imagen de muestra
      img_fondo_prob = fondo_pdf(img_hs) * 0.60 # Priori de fondo en la imagen de muestra
      
      # Aplicamos np.where sobre la imagen. Los pixeles detectados como fondo
      # se pintan de blanco, los demas se mantienen con el color original
      img_res = np.where(img_fondo_prob[..., None] > img_piel_prob[..., None], 
                     [255,255,255], img_bgr)
      
      # Se regresa al formato original
      img_res = np.array(img_res, dtype=np.uint8)
      # Se muestra el resultado de la clasificacion
      cv2.namedWindow('Seleccion de piel', cv2.WINDOW_NORMAL)
      cv2.resizeWindow('Seleccion de piel', 600, 400)
      cv2.imshow('Seleccion de piel', img_res)
      cv2.waitKey(0)
      cv2.destroyAllWindows()

    
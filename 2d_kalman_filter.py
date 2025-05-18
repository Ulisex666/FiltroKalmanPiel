import cv2 
import numpy as np
from skin_face_detection import get_face_roi

# Ubicacion de cada uno de los frames guardados
imgs_paths = ['imgs/frame1.jpg', 'imgs/frame2.jpg', 'imgs/frame3.jpg', 'imgs/frame4.jpg', 
              'imgs/frame5.jpg', 'imgs/frame6.jpg', 'imgs/frame7.jpg', 'imgs/frame8.jpg',
              'imgs/frame9.jpg', 'imgs/frame10.jpg', 'imgs/frame11.jpg', 'imgs/frame12.jpg', 
              'imgs/frame13.jpg', 'imgs/frame14.jpg']

# Inicializacion del filtro de Kalman. parte de OpenCV
# Buscamos predecir la posicion del centroide en el frame k.
# Se inicializa con 4 variables de estado y 2 de medicion.
# Variables de estado:
# Posicion X, posicion Y, velocidad X, velocidad Y
# Variables de medicion:
# Posicion X, posicion y
kalman_filter = cv2.KalmanFilter(4, 2)

# Matriz de transicion. Posicion x_k = x +dx, y_k = y + dy.
# La velocidad se asume constante 
kalman_filter.transitionMatrix = np.array(
    [[1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]], dtype=np.float32)

# Matriz de medida. Solo se nos da informacion de la posicion x y y
kalman_filter.measurementMatrix = np.array(
    [[1, 0, 0, 0],
    [0, 1, 0, 0]], dtype=np.float32)

# Matriz de ruido en el proceso. Se asume poco ruido independiente en
# cada variable
kalman_filter.processNoiseCov = np.eye(4, dtype=np.float32)*0.3
# Matriz de ruido para la medicion
kalman_filter.measurementNoiseCov = np.eye(2, dtype=np.float32)*0.1
# Matriz de covarianza para el error
kalman_filter.errorCovPost = np.eye(4, dtype=np.float32)

# Estado inicial del filtro de Kalman. Se asume que la cara esta en 
# el centro de la imagen
# El centro de la imagen (624,480) es (312, 240)
# Las medida Pre y Post son iguales dado que no se tienen medidas
kalman_filter.statePost = np.array([[312], [240], [0], [0]], dtype=np.float32)


for frame in imgs_paths:
    # Se elige la imagen y se aplica la funcion para detectar el rostro
    img = cv2.imread(frame)
    top_left, bottom_right, mid_point = get_face_roi(img)
    
    # Se usa el filtro de kalman para predecir el siguiente estado y se
    # guardan las coordenadas
    prediccion = kalman_filter.predict()
    x_predict = int(prediccion[0])
    y_predict = int(prediccion[1])

    # Se corrige la prediccion con la medicion obtenida
    kalman_filter.correct(np.array([mid_point[1], mid_point[0]], dtype=np.float32))
    
    # Se muestra la posicion corregida y la predecida
    # Posicion predicha en rojo
    cv2.circle(img, (y_predict, x_predict), 10, (0, 0, 255), 2)  
    # Posicion corregida en verde
    cv2.circle(img, mid_point, 10, (0, 255, 0), 2)  
    
    # Creamos la ROI  sobre la imagen y marcamos el centro
    cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 4)

    # Se muestra el resultado de la prediccion y la medicion 
    cv2.imshow(f'Filtro de kalman {frame}', img)
    cv2.waitKey(0)

cv2.destroyAllWindows()
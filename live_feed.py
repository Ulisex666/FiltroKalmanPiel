import cv2
import numpy as np
from skin_face_detection import get_face_roi

# Abrir webcam por defectro
cam = cv2.VideoCapture(0)

# Tamaño del frame
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Matriz de transicion. Posicion x_k = x +dx, y_k = y + dy.
# La velocidad se asume constante  
transition_matrix_2d = np.array(
    [[1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]], dtype=np.float32)

# Matriz de medida. Solo se nos da informacion de la posicion x y 
# y, no del movimiento
measurament_matrix_2d = np.array(
    [[1, 0, 0, 0],
    [0, 1, 0, 0]], dtype=np.float32)

# Matriz de ruido en el proceso. Se asume poco ruido independiente en
# cada variable
noise_process_matrix = np.eye(4, dtype=np.float32)*0.3

# Matriz de ruido para la medicion
noise_measurement_matrix = np.eye(2, dtype=np.float32)*0.1

# Matriz de covarianza para el error
error_cov = np.eye(4, dtype=np.float32)

# Inicializacion del filtro de Kalman. parte de OpenCV
# Buscamos predecir la posicion del centroide en el frame k.
# Se inicializa con 4 variables de estado y 2 de medicion.
# Variables de estado:
# Posicion X, posicion Y, velocidad X, velocidad Y
# Variables de medicion:
# Posicion X, posicion y
kalman_midpoint = cv2.KalmanFilter(4, 2)

# Alimentar los parametros al filtro de Kalman
kalman_midpoint.transitionMatrix = transition_matrix_2d
kalman_midpoint.measurementMatrix = measurament_matrix_2d
kalman_midpoint.processNoiseCov = noise_process_matrix
kalman_midpoint.measurementNoiseCov = noise_measurement_matrix
kalman_midpoint.errorCovPost = error_cov

# Estado inicial del filtro de Kalman. Se asume que la cara esta en 
# el centro de la imagen
# El centro de la imagen (624,480) es (312, 240)
# Las medida Pre y Post son iguales dado que no se tienen medidas
init = np.array([320, 240, 0, 0], dtype=np.float32)
kalman_midpoint.statePost = init

while True:
    # Se elige la imagen y se aplica la funcion para detectar el rostro
    ret, frame = cam.read()
    top_left, top_rigth, bottom_left, bottom_right, mid_point = get_face_roi(frame)
    
    # Se usa el filtro de kalman para predecir el siguiente estado y se
    # guardan las coordenadas
    prediccion = kalman_midpoint.predict()
    x_predict = int(prediccion[0])
    y_predict = int(prediccion[1])
    
    # Se corrige la prediccion con la medicion obtenida
    kalman_midpoint.correct(np.array([mid_point[1], mid_point[0]], dtype=np.float32))

    # Se muestra la posicion corregida y la predecida
    
    # Posicion predicha en rojo
    cv2.drawMarker(frame, (y_predict, x_predict), color=[0, 0, 255], markerType=cv2.MARKER_STAR, thickness=4, markerSize=20)
    cv2.putText(frame, f"Prediccion (x, y): ({x_predict}, {y_predict})", 
                (mid_point[0] + 10, mid_point[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0 , 0), 2)  
    
    # Posicion corregida en verde
    cv2.drawMarker(frame, mid_point, color=[0, 255, 0], markerType=cv2.MARKER_CROSS, thickness=4, markerSize=20)
    cv2.putText(frame, f"Medicion (x, y): ({mid_point[0]}, {mid_point[1]})", 
                (mid_point[0] + 10, mid_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255  ), 2)
 
    # Dibuja la region marcada como rostro
    points = np.array([top_left, bottom_left, bottom_right, top_rigth], np.int32)

    cv2.polylines(frame, [points], True, (0,255,0), 2)
    # cv2.drawMarker(frame, (mid_x, mid_y), color=[0,0,255], markerType=cv2.MARKER_SQUARE,
    #             thickness=4, markerSize=50)


    cv2.imshow('Camera', frame)
    # Press 'q' to exit the loop
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
#out.release()
cv2.destroyAllWindows()
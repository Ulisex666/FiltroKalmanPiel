import cv2
import numpy as np
from skin_face_detection import get_face_roi, formato_hsv

# Abrir webcam por defectro
cam = cv2.VideoCapture(0)

# Formato hsv es una bandera para decidir si se utiliza 
# ese formato en el proceso. Se importa de skin_face_detection para
# sincronizar ambos programas

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
noise_measurement_matrix = np.eye(2, dtype=np.float32)*0.3

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

# Se limita el número de frames a mostrar
frame_count = 0

while True:    
    
    # Se elige la imagen y se aplica la funcion para detectar el rostro
    ret, frame = cam.read()
    
    # Conversion a HSV. Debe sincronizarse manualmente con 
    if formato_hsv:
        frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        frame_hsv = frame_hsv[:,:,:2]
        top_left, top_rigth, bottom_left, bottom_right, mid_point = get_face_roi(frame_hsv)
    else:
        top_left, top_rigth, bottom_left, bottom_right, mid_point = get_face_roi(frame)
        
    # Nota: mid_point tiene las coordenadas "al reves", por lo que se maneja de manera diferente
    # a los demas puntos
    
    # Se usa el filtro de kalman para predecir el siguiente estado y se
    # guardan las coordenadas
    prediccion = kalman_midpoint.predict()
    x_predict = int(prediccion[0])
    y_predict = int(prediccion[1])
    v_x = float(prediccion[2])
    v_y = float(prediccion[3])
    # Se corrige la prediccion con la medicion obtenida
    correcion = kalman_midpoint.correct(np.array([mid_point[1], mid_point[0]], dtype=np.float32))
    x_corregido = int(correcion[0])
    y_corregido = int(correcion[1])
    vx_corregido = float(correcion[2])
    vy_corregido = float(correcion[3])
    
    # Se muestra la posicion corregida y la predecida, junto a la medicion
    
    # Medicion en amarillo
    cv2.drawMarker(frame, (mid_point[0], mid_point[1]), color=[0, 255, 255], markerType=cv2.MARKER_DIAMOND, thickness=4, markerSize=20)
    cv2.putText(frame, f"Medicion: x ={mid_point[0]}, y = {mid_point[1]}", 
                (400 , 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255 , 255), 2)  
    
    
    # Posicion predicha en rojo
    cv2.drawMarker(frame, (y_predict, x_predict), color=[0, 0, 255], markerType=cv2.MARKER_STAR, thickness=4, markerSize=20)
    cv2.putText(frame, f"Prediccion: x={y_predict}, y={x_predict}, v_x={v_y:.2f}, v_y={v_x:.2f}", 
                (230, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0 , 255), 2)

    
    # Posicion corregida en verde
    cv2.drawMarker(frame, (y_corregido, x_corregido), color=[0, 255, 0], markerType=cv2.MARKER_CROSS, thickness=4, markerSize=20)
    cv2.putText(frame, f"Correccion: x={y_corregido}, y={x_corregido}, v_x={vy_corregido:.2f}, v_y={vx_corregido:.2f}", 
                (230, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
 
    # Dibuja la region marcada como rostro
    points = np.array([top_left, bottom_left, bottom_right, top_rigth], np.int32)

    cv2.polylines(frame, [points], True, (255,0,0), 2)

    if frame_count % 15 == 0:
      cv2.imshow('Camera', frame)
    
    # Se actualiza el conteo de frames
    frame_count += 1
    
    # Cerrar loop principal con q
    if cv2.waitKey(1) == ord('q'):
        break

# Release the capture and writer objects
cam.release()
#out.release()
cv2.destroyAllWindows()
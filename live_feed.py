import cv2
import numpy as np
from skin_face_detection import get_face_roi

#
cam = cv2.VideoCapture(0)

# Get the default frame width and height
frame_width = int(cam.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT))

transition_matrix_2d = np.array(
    [[1, 0, 1, 0],
    [0, 1, 0, 1],
    [0, 0, 1, 0],
    [0, 0, 0, 1]], dtype=np.float32)

measurament_matrix_2d = np.array(
    [[1, 0, 0, 0],
    [0, 1, 0, 0]], dtype=np.float32)

noise_process_matrix = np.eye(4, dtype=np.float32)*0.3
noise_measurement_matrix = np.eye(2, dtype=np.float32)*0.1
error_cov = np.eye(4, dtype=np.float32)

kalman_midpoint = cv2.KalmanFilter(4, 2)

kalman_midpoint.transitionMatrix = transition_matrix_2d

kalman_midpoint.measurementMatrix = measurament_matrix_2d

kalman_midpoint.processNoiseCov = noise_process_matrix

kalman_midpoint.measurementNoiseCov = noise_measurement_matrix

kalman_midpoint.errorCovPost = error_cov

init = np.array([100, 100, 0, 0], dtype=np.float32)
kalman_midpoint.statePost = init
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (frame_width, frame_height))

while True:
    ret, frame = cam.read()
    top_left, bottom_right, mid_point = get_face_roi(frame)
    
    prediccion_midpoint = kalman_midpoint.predict()
    
    mid_x = int(prediccion_midpoint[0])
    mid_y = int(prediccion_midpoint[1])
    
    
    kalman_midpoint.correct(np.array([mid_point[1], mid_point[0]], dtype=np.float32))


    # Write the frame to the output file
    #out.write(frame)

    # Display the captured frame
    cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 4)
    cv2.drawMarker(frame, (mid_y, mid_x), color=[0,255,0], markerType=cv2.MARKER_CROSS,
                thickness=4, markerSize=50)
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
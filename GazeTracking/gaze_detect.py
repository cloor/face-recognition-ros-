"""
Demonstration of the GazeTracking library.
Check the README.md for complete documentation.
"""

import cv2
from gaze_tracking import GazeTracking
import time



gaze = GazeTracking()
webcam = cv2.VideoCapture(0)
threshold = 0.7
while True:
    # We get a new frame from the webcam
    _, frame = webcam.read()
    start_time = time.time()
    # We send this frame to GazeTracking to analyze it
    gaze.refresh(frame)
    frame = gaze.annotated_frame()
    
    text = ""
    
    if gaze.is_blinking():
        text = "Blinking"
    elif gaze.is_right():
        text = "Out of center"
    elif gaze.is_left():
        text = "Out of center"
    elif gaze.is_center():
        text = "Looking center"

    cv2.putText(frame, text, (90, 60), cv2.FONT_HERSHEY_DUPLEX, 1.6, (147, 58, 31), 2)

    left_pupil = gaze.pupil_left_coords()
    right_pupil = gaze.pupil_right_coords()
    FPS = 1.0 / (time.time() - start_time)
    cv2.putText(frame,'FPS: {:.1f}'.format(FPS),(10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(0,0,0))
    cv2.putText(frame, "Left pupil:  " + str(left_pupil), (90, 130), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)
    cv2.putText(frame, "Right pupil: " + str(right_pupil), (90, 165), cv2.FONT_HERSHEY_DUPLEX, 0.9, (147, 58, 31), 1)

    cv2.imshow("Demo", frame)

    if cv2.waitKey(1) == 27:
        break
   
webcam.release()
cv2.destroyAllWindows()

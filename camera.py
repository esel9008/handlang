import cv2
import sys

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera is not opened")
    sys.exit(1)

while True:
    res, frame = cap.read()

    if not res:
        print("Camera error")
        break
    
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllwindows()
cap.release()

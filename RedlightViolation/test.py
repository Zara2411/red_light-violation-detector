import cv2
import numpy as np

frame = cv2.imread("red.jpg")

# ROI around your traffic light (adjust!)
x, y, w, h = 120, 80, 40, 40
roi_img = frame[y:y+h, x:x+w]

# Convert to HSV
hsv = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)

# Super loose ranges for RED
lower1 = np.array([0, 10, 10])     # almost everything reddish
upper1 = np.array([15, 255, 255])
lower2 = np.array([160, 10, 10])
upper2 = np.array([180, 255, 255])

mask = cv2.inRange(hsv, lower1, upper1) | cv2.inRange(hsv, lower2, upper2)

# Debug display
cv2.imshow("ROI", roi_img)
cv2.imshow("HSV", hsv)
cv2.imshow("HSV Mask", mask)
cv2.waitKey(0)
cv2.destroyAllWindows()

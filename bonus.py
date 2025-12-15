import cv2
import os

path =  os.path.join('haar-cascade-files', 'haarcascade_lefteye_2splits.xml')
eye_cascade = cv2.CascadeClassifier(path)
img = cv2.imread("demo.png")
if img is True:
    print('yes')
else:
    print('no')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
eyes, numDetections = eye_cascade.detectMultiScale2(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
print(eyes)
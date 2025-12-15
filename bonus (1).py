import cv2
import os

leftpath =  os.path.join('haar-cascade-files', 'haarcascade_lefteye_2splits.xml')


lefteye_cascade = cv2.CascadeClassifier(leftpath)

img = cv2.imread("demo.png")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
lefteyes, numDetections = lefteye_cascade.detectMultiScale2(gray, scaleFactor=1.18, minNeighbors=4, minSize=(30, 30))

print(lefteyes)

for (x,y,h,w) in lefteyes:
    cv2.circle(img=img,center=(x+w//2,y+h//2),radius=20,color=(0,0,255),thickness=3)

cv2.imshow('scr',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


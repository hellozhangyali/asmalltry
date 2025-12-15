import cv2
# Load pre-trained cascade file
face_cascade = cv2.CascadeClassifier('C:/Users/ASUS/OneDrive/si100bp/haar-cascade-files/haarcascade_frontalface_default.xml')
# Read image and convert to grayscale
image = cv2.imread("demo.png")
# Convert to gray image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
print(faces)
# Draw rectangles around detected faces
for (x,y,z,w) in faces:
    cv2.rectangle(img=image,pt1=(x,y),pt2=(x+z,y+w),color=(255,0,255),thickness=3)
cv2.imshow('src',image)
# Display the result
cv2.waitKey(0)
cv2.destroyAllWindows()


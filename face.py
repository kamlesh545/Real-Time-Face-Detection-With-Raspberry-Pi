import cv2
from imutils.video import VideoStream  
import time

# Load Haarcascade classifier model for detect face
face_cascade = cv2.CascadeClassifier('./model/haarcascade_frontalface_default.xml')

# Using Pi Camera
PiCamera = True

# Set initial frame size.
frameSize = (1020, 720)

# Setup video stream
vs = VideoStream(src=0, usePiCamera=PiCamera, resolution=frameSize,framerate=32).start()

# Allow camera to setup.
time.sleep(2.0)
i = 0

while 1:
    # Read Video steram
    img = vs.read()
    # Convert frame into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Find faces in frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) == 0:
        # render a message on frame with no face detected
        cv2.putText(img, "NO FACE DETECTED", (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 4,cv2.LINE_AA)
    else:
        # render a message on frame with face detected
        cv2.putText(img, "FACE DETECTED", (340, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 4,cv2.LINE_AA)
        for (x,y,w,h) in faces:
            # Draw rectangle around every detected face
            cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)
            # Save the captured image with detected face
            cv2.imwrite(str(i)+'.png', img)
            i = i+1
    # Show result
    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()

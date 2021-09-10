import cv2
from simple_facerec import SimpleFacerec
from PIL import Image
import cv2
import face_recognition
import os
import dlib

# Encode faces from a folder
sfr = SimpleFacerec()
sfr.load_encoding_images("images/")

# Load Camera
cap = cv2.VideoCapture(0)

# dlib predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# dlib detector
detector = dlib.get_frontal_face_detector()

color = (0, 0, 200)

while True:
    ret, frame = cap.read()

    cropedFrame = []

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]

        try:
            roi = frame[y1-80: y2+50, x1-50: x2+50]
            croped = Image.fromarray(roi)
            cropedFrame.append(croped)
        except:
            pass


        # cv2.putText(frame, name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

    cv2.imshow("Frame", frame)
    
    if len(cropedFrame) == 2:
        color = (0, 255, 0)
        for i , img in enumerate(cropedFrame):
            img.save("test" + str(i) + ".jpg")

        image0 = cv2.imread("test0.jpg")
        image1 = cv2.imread("test1.jpg")
        gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

        face0 = detector(gray0, 1) #change frame size
        for face in face0:
            landmarks0 = predictor(gray0, face)

        face1 = detector(gray1, 2) #change frame size
        for face in face1:
            landmarks1 = predictor(gray1, face)

        for n in range(0, 68):
            try:
                x0 = landmarks0.part(n).x
                y0 = landmarks0.part(n).y
                cv2.circle(image0 , (x0, y0), 2, (240, 248, 255), -1)

                x1 = landmarks1.part(n).x
                y1 = landmarks1.part(n).y
                cv2.circle(image1 , (x1, y1), 2, (240, 248, 255), -1)
            except:
                pass

        cv2.imshow("img 0", image0)
        cv2.imshow("img 1", image1)
    else:
        color = (0, 0, 200)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()

os.remove("test0.jpg")
os.remove("test1.jpg")

cv2.destroyAllWindows()

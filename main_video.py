from typing import Counter
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

cap.set(3,1920)

# dlib predictor
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# dlib detector
detector = dlib.get_frontal_face_detector()

color = (0, 0, 200)

x = 0
a = ""

while True:
    ret, frame = cap.read()

    cropedFrame = []

    # Detect Faces
    face_locations, face_names = sfr.detect_known_faces(frame)
    for face_loc, name in zip(face_locations, face_names):
        y1, x2, y2, x1 = face_loc[0], face_loc[1], face_loc[2], face_loc[3]
        deltaY = abs(y1-y2)//2
        deltaX = abs(x1-x2)//2
        try:
            roi = frame[y1-deltaY: y2+deltaY, x1-deltaX: x2+deltaX]
            croped = Image.fromarray(roi)
            cropedFrame.append(croped)
        except:
            pass


        cv2.putText(frame, str(x) + ", A: " + a + ", Name: " + name,(x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 200), 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 4)

    cv2.imshow("Frame", frame)

    if len(cropedFrame) == 2:

        x += 1
        print(x)

        color = (255, 225, 0)

        if (x > 15):
            color = (0, 255, 0)
        
        if (x == 15):
            for i , img in enumerate(cropedFrame):
                # resized_img = img.resize((460, 500))
                # resized_img.save("test" + str(i) + ".jpg")

                img.save("test" + str(i) + ".jpg")

            # img.save("test" + str(i) + ".jpg")

            image0 = cv2.imread("test0.jpg")
            image1 = cv2.imread("test1.jpg")
            # gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2GRAY)
            # gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)

            #test encoding
            gray0 = cv2.cvtColor(image0, cv2.COLOR_BGR2RGB)
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
            try:
                img_encoding1 = face_recognition.face_encodings(gray0)[0]
                img_encoding2 = face_recognition.face_encodings(gray1)[0]
            except:
                pass
            # img_encoding1 = face_recognition.face_encodings(gray0)[0]
            # img_encoding2 = face_recognition.face_encodings(gray1)[0]
            result = face_recognition.compare_faces([img_encoding1], img_encoding2)
            a = str(result)
            print("Result: ", result)

            # face0 = detector(gray0, 1) #change frame size
            # for face in face0:
            #     landmarks0 = predictor(gray0, face)

            # face1 = detector(gray1, 2) #change frame size
            # for face in face1:
            #     landmarks1 = predictor(gray1, face)

            # for n in range(0, 68):
            #     try:
            #         x0 = landmarks0.part(n).x
            #         y0 = landmarks0.part(n).y
            #         cv2.circle(image0 , (x0, y0), 2, (240, 248, 255), -1)

            #         x1 = landmarks1.part(n).x
            #         y1 = landmarks1.part(n).y
            #         cv2.circle(image1 , (x1, y1), 2, (240, 248, 255), -1)
            #     except:
            #         pass
        
            # cv2.imshow("img 0", image0)
            # cv2.imshow("img 1", image1)
        
    elif len(cropedFrame) == 1:
        x = 0
        a = ""
        color = (0, 0, 200)

    else:
        try:
            os.remove("test0.jpg")
            os.remove("test1.jpg")
        except:
            pass

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()

try:
    os.remove("test0.jpg")
    os.remove("test1.jpg")
except:
    pass

cv2.destroyAllWindows()

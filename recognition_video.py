import pickle
import cv2
import os

import face_detect

names = pickle.load(open('labels.pkl','rb'))

model = cv2.face.LBPHFaceRecognizer_create()
model.read('employees_model.yml')

def face_rec_video(videopath):
    faceCascade = cv2.CascadeClassifier("opencv-files/haarcascade_frontalface_alt.xml")
    video_capture = cv2.VideoCapture(videopath)
    # pTime = 0
    while True:
        ret, frame = video_capture.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            label, confidence = model.predict(gray[y:y+w, x:x+h])
            if confidence > 60:
                cv2.putText(frame, names[label], (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            0.75, (0, 255, 0), 2)


        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()

face_rec_video(0)
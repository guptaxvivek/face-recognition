import cv2
import os
import pickle

dirs = os.listdir('training-data')
for dir_name in dirs:
    if not dir_name.startswith("s"):
        continue;
    label = int(dir_name.replace("s", ""))

try:
    with open('labels.pkl','rb') as f:
        labels = pickle.load(f)
except FileNotFoundError:
    labels = {}
    print("Not Found")
    label = 0

# print(labels)
def start_capture(name):
        path = "training-data/s" + str(label+1)
        num_of_images = 0
        detector = cv2.CascadeClassifier("opencv-files/haarcascade_frontalface_alt.xml")
        try:
            os.makedirs(path)
        except:
            print('Directory Already Created')
        vid = cv2.VideoCapture(0)
        # vid.set(cv2.CAP_PROP_FRAME_WIDTH,1280)
        # vid.set(cv2.CAP_PROP_FRAME_HEIGHT,720)
        while True:

            ret, img = vid.read()
            new_img = None
            grayimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            face = detector.detectMultiScale(image=grayimg, scaleFactor=1.1, minNeighbors=5)
            for x, y, w, h in face:
                new_img = img[y:y+h, x:x+w]
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
                cv2.putText(img, "Face Detected", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))
                cv2.putText(img, str(str(num_of_images) + " images captured"), (x, y + h + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255))

            cv2.imshow("FaceDetection", img)
            key = cv2.waitKey(1) & 0xFF


            try :
                cv2.imwrite(str(path+"/"+str(num_of_images)+name+".jpg"), new_img)
                num_of_images += 1
            except :
                pass

            if key == ord("q") or key == 27 or num_of_images > 150:
                labels[label+1] = name
                with open('labels.pkl','wb') as f:
                    pickle.dump(labels, f, protocol=pickle.HIGHEST_PROTOCOL)
                break
        cv2.destroyAllWindows()
        return num_of_images

start_capture('Vivek Gupta')
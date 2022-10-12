import cv2
import face_recognition
import pickle
import numpy as np
import sys, os, glob
import dlib
import time
from datetime import datetime
from datetime import date
import threading
import pandas as pd


# cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
today = date.today()
# if os.path.exists("{}.csv".format(today)):
#     data = pd.read_csv("{}.csv".format(today))
# else:
data = pd.read_csv("data.csv")

class FaceRecognition(object):
    
    def face_recognition(self):
        while(self.cap.isOpened()):
            ret, frame = self.cap.read()
            if ret:
                # frame = cv2.resize(frame, (1080,720))
                # gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                rgb_small_frame = small_frame[:, :, ::-1]
                faces = detector(small_frame, 1)

                with open('face_encoding.pickle', 'rb+') as handle:
                    load_data = pickle.load(handle)

                
                # faces = facec.detectMultiScale(small_frame, 1.3, 5)
                # print(faces)
                today = date.today()

                if not os.path.isdir(str(today)):
                    os.mkdir(str(today))
                if len(faces) > 0:
                    known_face_encodings = load_data["face_encodings"]
                    known_face_names = load_data["user_names"]
                    # print(len(known_face_encodings),len(known_face_names))
                    # print(known_face_names)
                    face_locations = face_recognition.face_locations(rgb_small_frame)
                    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                    face_names = []
                    for face_encoding in face_encodings:
                        # See if the face is a match for the known face(s)
                        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                        name = "Unknown"

                        # # If a match was found in known_face_encodings, just use the first one.
                        # if True in matches:
                        #     first_match_index = matches.index(True)
                        #     name = known_face_names[first_match_index]

                        # Or instead, use the known face with the smallest distance to the new face
                        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                        best_match_index = np.argmin(face_distances)
                        if matches[best_match_index]:
                            name = known_face_names[best_match_index]    
                            now = datetime.now()
                            print(now)
                            data.loc[len(data.index)] = [name, str(now)] 
                            # data.to_csv("{}.csv".format(today), index=False)
                            data.to_csv("data.csv", index=False)
                            
                        else:
                            count = len(known_face_names) + 1
                            known_face_encodings.append(face_encoding)
                            known_face_names.append(str(count))
                            store_data = {"face_encodings" : known_face_encodings,"user_names": known_face_names}
                            with open(f'face_encoding.pickle', 'wb+') as handle:
                                pickle.dump(store_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


                        print(name)
                        for d in faces:
                            crop_img = small_frame[d.top():d.bottom(),d.left():d.right()]
                            cv2.imwrite("{}/{}.jpg".format(str(today), name), crop_img)
                # cv2.imshow('frame', frame)
                # if cv2.waitKey(20) & 0xFF == ord('q'):
                #     break
                _, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()
        # cap.release()
        # cv2.destroyAllWindows()
    
    def __init__(self, stream_link):
        self.cap = cv2.VideoCapture(stream_link) 
        # t = threading.Thread(target=self.face_recognition)
        # t.start()
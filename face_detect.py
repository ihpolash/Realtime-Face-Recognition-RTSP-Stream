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
data = pd.read_csv("data.csv")

    
def face_detect(image_path):
    frame = cv2.imread(image_path)
    # frame = cv2.resize(frame, (1080,720))
    # gray_fr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    # rgb_small_frame = small_frame[:, :, ::-1]
    faces = detector(frame, 1)

    with open('face_encoding.pickle', 'rb+') as handle:
        load_data = pickle.load(handle)

    known_face_encodings = load_data["face_encodings"]
    known_face_names = load_data["user_names"]
    # print(len(known_face_encodings),len(known_face_names))
    # print(known_face_names)
    face_locations = face_recognition.face_locations(frame)
    print(face_locations)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    face_names = []
    name = "Unknown"
    for face_encoding in face_encodings:
        # See if the face is a match for the known face(s)
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        # # If a match was found in known_face_encodings, just use the first one.
        # if True in matches:
        #     first_match_index = matches.index(True)
        #     name = known_face_names[first_match_index]

        # Or instead, use the known face with the smallest distance to the new face
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            entry_list = data[data['name'] == name]
            last_entry = entry_list['datetime'].max()
            
        else:
            name = "Unknown"


        print(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        #count no of detected faces
        cv2.putText(frame, str(len(faces)), (50, 50), font, 1.0, (255, 255, 255), 1)
        print(len(faces))

    return name, frame, last_entry
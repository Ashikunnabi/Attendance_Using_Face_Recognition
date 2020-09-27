import cv2
import numpy as np
import face_recognition
import os
import pickle
import json
from datetime import datetime

pickled_filename = "picklefile"
user_names, encoded_user_data = [], []


def read_pickle_file():
    data = []
    with open(pickled_filename, 'rb') as fr:
        try:
            while True:
                data.append(pickle.load(fr))
        except EOFError:
            pass
    return data


for i in read_pickle_file():
    user_names.append(list(i.keys())[0])
    encoded_user_data.append(list(i.values())[0])


def mark_attendance(name):
    with open('Attendance.csv', 'r+') as f:
        my_data_list = f.readlines()
        name_list = []
        for line in my_data_list:
            entry = line.split(',')
            name_list.append(entry[0])
        if name not in name_list:
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.writelines('\n{0},{1}'.format(name, dt_string))


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgS)
    encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encoded_user_data, encodeFace)
        faceDis = face_recognition.face_distance(encoded_user_data, encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = user_names[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            mark_attendance(name)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)

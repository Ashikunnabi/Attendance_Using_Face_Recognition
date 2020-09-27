import cv2
import numpy as np
import face_recognition
import os
import pickle
from pathlib import Path
from datetime import datetime

new_image_path = 'new_images'
pickled_image_path = 'pickled_images'
pickled_filename = "picklefile"
full_image_names, images, user_names = [], [], []
BASE_DIR = Path().absolute()
unpickled_images_dir = os.listdir(new_image_path)

for cl in unpickled_images_dir:
    curImg = cv2.imread('{0}/{1}'.format(new_image_path, cl))
    images.append(curImg)
    full_image_names.append(cl)
    user_names.append(os.path.splitext(cl)[0])


def find_encodings(images):
    encode_list = []
    for image in images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(image)[0]
        encode_list.append(encode)
    return encode_list


def pickle_image_data(full_image_names, image_names, image_encodings):
    for full_image_name, image_name, image_encoding in zip(full_image_names, image_names, image_encodings):
        with open(pickled_filename, 'ab+') as fp:
            data = {image_name: image_encoding}
            pickle.dump(data, fp)
        # moving pickled images to pickled_image folder
        Path("{0}/{1}/{2}".format(BASE_DIR, new_image_path, full_image_name)).rename(
            "{0}/{1}/{2}".format(BASE_DIR, pickled_image_path, full_image_name))


encodeListKnown = find_encodings(images)
pickle_image_data(full_image_names, user_names, encodeListKnown)
print('Registration Complete')

__author__ = 'Ivanov Andrew'

import numpy as np
import cv2
import cv2 as cv

folder = "input"
prefix = "rdimage"
extension = ".ppm"

def get_file_name (index):
    return folder + "/" + prefix + ".00" + str(index) + extension

class Camera:
    def __init__(self, camera_matrix, distortion, rotation, translation):
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.rotation = rotation
        self.translation = translation

def parse_vector(str):

    split_res = str.split()
    res = []

    for x in split_res:
        res.append(float(x))

    return res


def parse_matrix(one, two, three):

    fst_line = parse_vector(one)
    snd_line = parse_vector(two)
    thr_line = parse_vector(three)

    return np.array([fst_line, snd_line, thr_line])

def parse_camera_data (index):

    file_name = get_file_name(index) + ".camera"

    file = open(file_name, "r")

    camera_matrix = parse_matrix(file.readline(), file.readline(), file.readline())
    distortion = parse_vector(file.readline())
    rotation_matrix = parse_matrix(file.readline(), file.readline(), file.readline())
    translation_vector = parse_vector(file.readline())

    return Camera(camera_matrix, distortion, rotation_matrix, translation_vector)

fst_image = cv2.imread(get_file_name(0), cv2.CV_LOAD_IMAGE_COLOR)
fst_camera = parse_camera_data(0)

snd_image = cv2.imread(get_file_name(1), cv2.CV_LOAD_IMAGE_COLOR)
snd_camera = parse_camera_data(1)

#cv.stereoRectify(fst_camera.camera_matrix, fst_camera.distortion, snd_camera.camera_matrix, snd_camera.distortion, )


__author__ = 'Ivanov Andrew'

import numpy as np
from numpy.linalg import inv
import cv2

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
    res = np.zeros(3)

    for i in range(0, 3):
        res[i] = (float(split_res[i]))

    return res


def parse_matrix(one, two_camera, three):

    fst_line = parse_vector(one)
    snd_line = parse_vector(two_camera)
    thr_line = parse_vector(three)

    return np.array([fst_line, snd_line, thr_line])


def parse_camera_data (index):

    file_name = get_file_name(index) + ".camera"

    file = open(file_name, "r")
    lines = file.readlines()

    K_matrix = parse_matrix(lines[0], lines[1], lines[2])
    #file.readline()
    #distortion = parse_vector(file.readline())
    distortion = np.zeros(5)
    R_matrix = np.transpose(parse_matrix(lines[4], lines[5], lines[6]))
    #t = - R_1 * t_1
    T_vector = - np.dot(R_matrix, parse_vector(lines[7]))

    file.close()

    return Camera(K_matrix, distortion, R_matrix, T_vector)


def calculate_R_matrix(one, two):

    R1 = one.rotation
    R2 = two.rotation
    # R_2 * R_1^-1
    return np.dot(R2, np.linalg.inv(R1))


def calculate_T_vector(one, two):

    R1, R2 = one.rotation, two.rotation
    t1, t2 = one.translation, two.translation

    # - R_2 * R_1^-1 * t_1 + t_2
    res = np.dot(-R2, np.dot(np.linalg.inv(R1), t1)) + t2
    return res


def rectify(image, camera, R1, P1):

    height, width = image.shape
    size = (width, height)

    C, D = camera.camera_matrix, camera.distortion

    #params:
    # cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type
    map_x, map_y = cv2.initUndistortRectifyMap(C, D, R1, P1, size, cv2.CV_8UC1)

    new_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    return new_image


def get_image_and_camera(index):

    filename = get_file_name(index)
    image = cv2.imread(filename, cv2.CV_8UC1)
    camera = parse_camera_data(index)

    return image, camera

fst_image, fst_camera = get_image_and_camera(0)
snd_image, snd_camera = get_image_and_camera(1)

R = calculate_R_matrix(fst_camera, snd_camera)
T = calculate_T_vector(fst_camera, snd_camera)

height, width = fst_image.shape
size = (width, height)

#params:
#cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(fst_camera.camera_matrix, fst_camera.distortion,\
                                                  snd_camera.camera_matrix, snd_camera.distortion,\
                                                  size, R, T, None, 0)

fst_rectified_image = rectify(fst_image, fst_camera, R1, P1)
snd_rectified_image = rectify(snd_image, snd_camera, R2, P2)

cv2.imwrite("rect1.png", fst_rectified_image)
cv2.imwrite("rect2.png", snd_rectified_image)
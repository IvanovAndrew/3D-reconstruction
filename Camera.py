__author__ = 'Ivanov Andrew'

import numpy as np


class Camera:
    def __init__(self, camera_matrix, distortion, rotation, translation):
        self.camera_matrix = camera_matrix
        self.distortion = distortion
        self.rotation = rotation
        self.translation = translation

    @classmethod
    def from_file_name(cls, filename):

        file = open(filename, "r")
        lines = file.readlines()

        K_matrix = parse_matrix(lines[0], lines[1], lines[2])

        distortion = np.zeros(5)
        R_matrix = np.transpose(parse_matrix(lines[4], lines[5], lines[6]))
        #t = - R_1 * t_1
        T_vector = - np.dot(R_matrix, parse_vector(lines[7]))

        file.close()

        return cls(K_matrix, distortion, R_matrix, T_vector)

    def get_fx(self):
        return self.camera_matrix[0, 0]


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
__author__ = 'Ivanov Andrew'

import os
import numpy as np
import cv2
import Camera


def read_image_and_camera(index):

    filename = "input/rdimage.00" + str(index) + ".ppm"
    image = cv2.imread(filename, cv2.CV_8UC1)

    camera_filename = filename + ".camera"
    camera = Camera.Camera.from_file_name(camera_filename)

    return image, camera


def calculate_rotation(one, two):

    R1 = one.rotation
    R2 = two.rotation
    # R_2 * R_1^-1
    return np.dot(R2, np.linalg.inv(R1))


def calculate_translation(one, two):

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


def get_disparity(left_rectified, right_rectified):

    min_disparity = 400
    num_of_disparities = 832 - min_disparity
    window_size = 11
    stereo = cv2.StereoSGBM\
            (minDisparity=min_disparity,
             numDisparities=num_of_disparities,
             SADWindowSize=window_size)

    disparity = stereo.compute(left_rectified, right_rectified)
    return disparity.astype(np.float32)/16.0 - min_disparity



fst_image, fst_camera = read_image_and_camera(0)
snd_image, snd_camera = read_image_and_camera(1)

R = calculate_rotation(fst_camera, snd_camera)
T = calculate_translation(fst_camera, snd_camera)

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

disparity = get_disparity(fst_rectified_image, snd_rectified_image)

print(disparity.shape)

cv2.imwrite("depthmap.bmp", disparity)


#http://docs.opencv.org/2.4/modules/core/doc/operations_on_arrays.html#normalize
#cv2.imshow("depthmap", resDisparity)
#cv2.waitKey()
#temp = cv2.reprojectImageTo3D(disparity, Q)

# file = open("3dpict.obj", "w")
# file.write(line)
# file.close()

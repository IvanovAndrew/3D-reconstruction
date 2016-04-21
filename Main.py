__author__ = 'Ivanov Andrew'

import os
import datetime
import numpy as np
import cv2
import Camera

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


class DisparityParams:

    def __init__(self, min_disparity, max_disparity, window_size):

        self.min_disparity = min_disparity
        self.max_disparity = max_disparity
        self.num_of_disparity = max_disparity - min_disparity
        self.window_size = window_size


class Volume:

    def __init__(self, x_range, y_range, z_range):

        x_min, x_max = x_range
        y_min, y_max = y_range
        z_min, z_max = z_range
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.z_min = z_min
        self.z_max = z_max


def write_ply(fn, verts):

    print 'start write ' + fn + " " + str(datetime.datetime.now())
    with open(fn, 'wb') as f:

        f.write((ply_header % dict(vert_num=len(verts))).encode('utf-8'))
        np.savetxt(f, verts, fmt='%f %f %f %d %d %d ')

    print 'finish write ' + fn + " " + str(datetime.datetime.now())


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


def get_rectified_image(image, camera, R1, P1):

    height, width = image.shape
    size = (width, height)

    C, D = camera.camera_matrix, camera.distortion

    #params:
    # cameraMatrix, distCoeffs, R, newCameraMatrix, size, m1type
    map_x, map_y = cv2.initUndistortRectifyMap(C, D, R1, P1, size, cv2.CV_8UC1)

    new_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)

    return new_image


def rectify(fst_camera, fst_image, snd_camera, snd_image):

    R = calculate_rotation(fst_camera, snd_camera)
    T = calculate_translation(fst_camera, snd_camera)

    height, width = fst_image.shape
    size = (width, height)

    #params:
    #cameraMatrix1, distCoeffs1, cameraMatrix2, distCoeffs2, imageSize, R, T
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(fst_camera.camera_matrix, fst_camera.distortion,\
                                                      snd_camera.camera_matrix, snd_camera.distortion,\
                                                      size, R, T, None, 0)

    fst_rectified_image = get_rectified_image(fst_image, fst_camera, R1, P1)
    snd_rectified_image = get_rectified_image(snd_image, snd_camera, R2, P2)

    return fst_rectified_image, R1, snd_rectified_image, Q


def get_disparity(left_rectified, right_rectified, params):

    min_disparity = params.min_disparity
    num_of_disparities = params.num_of_disparity
    window_size = params.window_size
    P1 = 8 * 2 * window_size * window_size
    P2 = 32 * 2 * window_size * window_size
    stereo = cv2.StereoSGBM\
            (minDisparity=min_disparity,
             numDisparities=num_of_disparities,
             SADWindowSize=window_size,
             P1=P1,
             P2=P2,
             speckleWindowSize=200,
             speckleRange=1)

    disparity = stereo.compute(left_rectified, right_rectified)
    return disparity.astype(np.float32)/16.0


def get_points_cloud(filename, disparity, Q):

    left = cv2.imread(filename)

    print Q
    points = cv2.reprojectImageTo3D(disparity, Q)
    colors = cv2.cvtColor(left, cv2.COLOR_BGR2RGB)
    mask = disparity > disparity.min()
    out_points = points[mask]
    out_colors = colors[mask]

    verts = out_points.reshape(-1, 3)
    colors = out_colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])

    return verts


def filter_points(cloud, volume):

    cloud = cloud[volume.x_min <= cloud[:,0]]
    cloud = cloud[cloud[:,0] <= volume.x_max]
    cloud = cloud[volume.y_min <= cloud[:,1]]
    cloud = cloud[cloud[:,1] <= volume.y_max]
    cloud = cloud[cloud[:,2] <= volume.z_max]
    cloud = cloud[volume.z_min <= cloud[:,2]]

    return cloud


def get_cloud(image, disparity, volume, Q, plyname):

    cloud = get_points_cloud(image, disparity, Q)
    cloud = filter_points(cloud, volume)

    write_ply(plyname, cloud)

    return cloud


def to_world_coordinates(points, rotation_to_camera, translation, rotation_to_rectify):

    res = []
    for i in range(0,len(points)):
        camera_point = np.dot(np.linalg.inv(rotation_to_rectify), points[i][0])
        world_point = np.dot(np.linalg.inv(rotation_to_camera), camera_point - translation)
        res.append(world_point)
        res.append(points[i][1])

    res = np.array(res)
    res = res.reshape(-1, 6)

    return res


def get_global_coordinates(left_index, right_index, disparity_params, volume_params):

    image_0, camera_0 = read_image_and_camera(left_index)
    image_1, camera_1 = read_image_and_camera(right_index)

    rectified_0, R_rect, rectified_1, Q = rectify(camera_0, image_0, camera_1, image_1)

    rect1 = "rect" + str(left_index) + str(right_index) + "_" + str(left_index) + ".png"
    rect2 = "rect" + str(left_index) + str(right_index) + "_" + str(right_index) + ".png"
    cv2.imwrite(rect1, rectified_0)
    cv2.imwrite(rect2, rectified_1)


    disparity = get_disparity(rectified_0, rectified_1, disparity_params)

    depthname = "depthmap" + str(left_index) + str(right_index) + ".bmp"
    cv2.imwrite(depthname, disparity - disparity_params.min_disparity)

    local_ply_name = "local " + str(left_index) + str(right_index) + ".ply"
    verts = get_cloud(rect1, disparity, volume_params, Q, local_ply_name)
    verts = verts.reshape(-1, 2, 3)
    global_coords = to_world_coordinates(verts, camera_0.rotation, camera_0.translation, R_rect)

    global_ply_name = "global " + str(left_index) + str(right_index) + ".ply"
    write_ply(global_ply_name, global_coords)

    return global_coords


def get01GlobalCoords():

    print "01"
    disparity_params = DisparityParams(min_disparity=400, max_disparity=832, window_size=11)
    volume_params = Volume((-50, 500), (-500, 500), (-500, 400))

    return get_global_coordinates(0, 1, disparity_params, volume_params)


# def get32GlobalCoords():
#     print "32"
#     ### 1845 -  90 ; 1960 - 203 ; 2256-505; 2637 - 1081; 2641 - 1033; 2659 - 1083
#     ### 2791 - 1409; 2957 - 1523; 2487 - 905; 1829 - 61; 1881 - 119; 2553 - 883
#     ### 2669 - 1177;
#     disparity_params = DisparityParams(min_disparity=1360, max_disparity=1776, window_size=11)
#     volume_params = Volume((-500, 500), (-500, 500), (-500, 500))
#
#     verts, camera, R_rect = get_cloud(3, 2, disparity_params, volume_params)
#     verts = verts.reshape(-1, 2, 3)
#     globalCoords = to_world_coordinates(verts, camera.rotation, camera.translation, R_rect)
#     write_ply("32global.ply", globalCoords)
#     return globalCoords


# def get45GlobalCoords():
#     print "45"
#     # 1565 - 191; 1547 - 201; 1541 - 205; 1831 - 499; 1945 - 519;
#     # 1745 - 565; 1487 - 240; 2141 - 925; 1831 - 129; 1755 - 71
#     disparity_params = DisparityParams(min_disparity=992, max_disparity=1712, window_size=17)
#     volume_params = Volume((0, 500), (-500, 500), (-500, 400))
#
#     verts, camera, R_rect = get_cloud(4, 5, disparity_params, volume_params)
#     verts = verts.reshape(-1, 2, 3)
#     globalCoords = to_world_coordinates(verts, camera.rotation, camera.translation, R_rect)
#     write_ply("45global.ply", globalCoords)
#     return globalCoords


def get56GlobalCoords():
    print "56"
    # 866 - 71; 1236 - 280; 1158 - 189; 1614 - 653; 1484 - 635; 1126 - 220; 1088 - 263
    # 2126 - 1325; 2252 - 1411; 2506 - 1863; 2478 - 1849; 2448 - 1829; 2312 - 1721
    # 2392 - 1803; 2260 - 1667; 2202 - 1628
    # 2485 - 1878; 2825 - 2101
    disparity_params = DisparityParams(min_disparity=576, max_disparity=992, window_size=11)
    volume_params = Volume((0, 200), (-500, 500), (-100, 300))

    return get_global_coordinates(5, 6, disparity_params, volume_params)

print datetime.datetime.now()
coords1 = get01GlobalCoords()
#get32GlobalCoords()
# coords45 = get45GlobalCoords()
print datetime.datetime.now()
coords56 = get56GlobalCoords()
print datetime.datetime.now()
# print "32"
# disparity_params = DisparityParams(min_disparity=1376, max_disparity=1872, window_size=3)
# volume_params = Volume((-100, 100), (-100, 100), (-100, 100))
# cloud32 = get_cloud(3, 2, disparity_params, volume_params)
import numpy as np

f_rgb_xyz = lambda x: ((x + 0.055) / 1.055) ** 2.4 if x > 0.04045 else x / 12.92
f_xyz_lab = lambda x: x ** (1/3) if x > (6/29) ** 3 else (1/3) * ((29/6)**2) * x + (6/29) 

def rgb_to_xyz(image_rgb):
    image_xyz = image_rgb.copy().astype(np.float64)
    image_xyz /= 255
    for i in range(3):
        img = image_xyz[:, :, i].flatten()
        img = np.array(list(map(f_rgb_xyz, img)))
        img = img.reshape(image_xyz[:, :, i].shape)
        image_xyz[:, :, i] = img
    # image_xyz *= 100.0
    xyz = [
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ]
    img_xyz = image_xyz.copy()
    for i in range(3):
        image_xyz[:, :, i] = img_xyz[:, :, 0] * xyz[i][0] + img_xyz[:, :, 1] * xyz[i][1] + img_xyz[:, :, 2] * xyz[i][2]
        image_xyz[:, :, i] *= 255
    return image_xyz.astype(int)

def xyz_to_lab(image_xyz):
    image_lab = image_xyz.copy().astype(np.float64)
    xyz_n = [95.047, 100.00, 108.883]
    for i in range(3):
        img = image_lab[:, :, i].flatten()
        img /= xyz_n[i]
        img = np.array(list(map(f_xyz_lab, img)))
        img = img.reshape(image_lab[:, :, i].shape)
        image_lab[:, :, i] = img
    img_lab = image_lab.copy()
    image_lab[:, :, 0] = img_lab[:, :, 1] - 16
    image_lab[:, :, 1] = (img_lab[:, :, 0] - img_lab[:, :, 1]) * 500
    image_lab[:, :, 2] = (img_lab[:, :, 1] - img_lab[:, :, 2]) * 200
    image_lab[:, :, 0] = image_lab[:, :, 0] / 100 * 255
    image_lab[:, :, 1] = (image_lab[:, :, 1] + 128)
    image_lab[:, :, 2] = (image_lab[:, :, 2] + 128)
    return image_lab.astype(int)

def binary(image_gray, umbral):
    binary_image = image_gray.copy()
    for i in range(len(image_gray)):
        for j in range(len(image_gray[i])):
            if image_gray[i][j] > umbral:
                binary_image[i][j] = 255
            else:
                binary_image[i][j] = 0
    return binary_image
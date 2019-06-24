'''

Image Sensor Vacuum Cleaner.

Author:
C. Meyer
'''

import os
import imageio
import numpy as np
from numba import jit

images_list = []
files_list = []
files_number = 0

print("ISVaC")

# Read file names
for file in os.listdir("./data"):
    if file.endswith(".JPG"):
        files_list.append(os.path.join("./data", file))
        files_number += 1

# Read images
print("Read images")
i = 1
for filename in files_list:
    print(str(i), "/", str(files_number), os.path.join("./data", file))
    i += 1
    image = imageio.imread(filename)
    images_list.append(image)

# convert images list in numpy array for numba
images_list = np.array(images_list, dtype=np.uint8)


# TODO : add a verification to check that all images are the same shape


# Create the mask
# TODO : better solution than a threshold on the three channel variance
@jit(nopython=True)
def compute_mask(images, threshold):
    mask = np.zeros((images.shape[1], images.shape[2]), dtype=np.uint8)
    variance_map = np.zeros(images[0].shape, dtype=np.int32)

    for i in range(0, images.shape[1]):
        for j in range(0, images.shape[2]):
            c0 = np.zeros((images.shape[0]), dtype=np.int32)
            c1 = np.zeros((images.shape[0]), dtype=np.int32)
            c2 = np.zeros((images.shape[0]), dtype=np.int32)
            for k in range(0, images.shape[0]):
                c0[k] = images[k][i][j][0]
                c1[k] = images[k][i][j][1]
                c2[k] = images[k][i][j][2]
            variance_map[i][j][0] = c0.var()
            variance_map[i][j][1] = c1.var()
            variance_map[i][j][2] = c2.var()

            if np.sum(variance_map[i][j]) < threshold:
                mask[i][j] = 1

    return mask


print("Search for common pixels, and add them to the mask)")
mask = compute_mask(images_list, 1000).astype(bool)


# This is completely specific to my images
def mask_remove_date_zone(mask):
    for i in range(1798, 1865):
        for j in range(1916, 2429):
            mask[i][j] = False

    return mask


print("Removing date zone from mask")
mask = mask_remove_date_zone(mask)


def mask_dilation(mask):
    new_mask = np.zeros(mask.shape)

    for i in range(1, mask.shape[0]-1):
        for j in range(1, mask.shape[1]-1):
            if mask[i][j]:
                new_mask[i][j] = True
                new_mask[i][j+1] = True
                new_mask[i][j-1] = True
                new_mask[i+1][j] = True
                new_mask[i+1][j+1] = True
                new_mask[i+1][j-1] = True
                new_mask[i-1][j] = True
                new_mask[i-1][j+1] = True
                new_mask[i-1][j-1] = True

    return new_mask


print("Mask dilatation")
for i in range(0, 3):
    mask = mask_dilation(mask)

print("Write the dirt mask")
mask_image = (mask*255).astype("uint8")
imageio.imwrite("MASK.JPG", mask_image)


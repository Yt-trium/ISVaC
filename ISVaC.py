'''

Image Sensor Vacuum Cleaner.

Author:
C. Meyer
'''

import os
import imageio
import numpy as np
from numba import jit

# User parameters
user_threshold = 750
user_dilatation = 4


if not os.path.exists("./masks_" + str(user_threshold) + "_" + str(user_dilatation)):
    os.mkdir("./masks_" + str(user_threshold) + "_" + str(user_dilatation))
if not os.path.exists("./results_" + str(user_threshold) + "_" + str(user_dilatation)):
    os.mkdir("./results_" + str(user_threshold) + "_" + str(user_dilatation))


images_list = []
files_list = []
file_names_list = []
files_number = 0

print("ISVaC")

# Read file names
for file in os.listdir("./data"):
    if file.endswith(".JPG"):
        files_list.append(os.path.join("./data", file))
        file_names_list.append(file)
        files_number += 1

# Read images
print("Read images")
cpt = 1
for filename in files_list:
    print(str(cpt), "/", str(files_number), os.path.join("./data", file))
    cpt += 1
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
mask = compute_mask(images_list, user_threshold).astype(bool)


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
for cpt in range(0, user_dilatation):
    mask = mask_dilation(mask)

print("Write the dirt mask")
imageio.imwrite(os.path.join("./results_" + str(user_threshold) + "_" + str(user_dilatation), "MASK.JPG"), (mask*255).astype("uint8"))


def expand_components(local_mask, component, i, j):
    if local_mask[i + 1][j]:
        local_mask[i + 1][j] = False
        component.append((i + 1, j))
        local_mask, component = expand_components(local_mask, component, i + 1, j)
    if local_mask[i - 1][j]:
        local_mask[i - 1][j] = False
        component.append((i - 1, j))
        local_mask, component = expand_components(local_mask, component, i - 1, j)
    if local_mask[i][j + 1]:
        local_mask[i][j + 1] = False
        component.append((i, j + 1))
        local_mask, component = expand_components(local_mask, component, i, j + 1)
    if local_mask[i][j - 1]:
        local_mask[i][j - 1] = False
        component.append((i, j - 1))
        local_mask, component = expand_components(local_mask, component, i, j - 1)

    return local_mask, component


def get_components(mask):

    components = []
    components_min_x = []
    components_max_x = []
    components_min_y = []
    components_max_y = []

    local_mask = np.copy(mask)
    c = 0

    for i in range(0, local_mask.shape[0]):
        for j in range(0, local_mask.shape[1]):
            if local_mask[i][j]:
                local_mask[i][j] = False
                component = [(i, j)]
                local_mask, component = expand_components(mask, component, i, j)

                max_x = 0
                max_y = 0
                min_x = 10e10
                min_y = 10e10

                for k in range(0, len(component)):
                    min_x = min(component[k][0], min_x)
                    max_x = max(component[k][0], max_x)
                    min_y = min(component[k][1], min_y)
                    max_y = max(component[k][1], max_y)

                components.append(component)
                components_min_x.append(min_x)
                components_max_x.append(max_x)
                components_min_y.append(min_y)
                components_max_y.append(max_y)

                c += 1

    return components, components_min_x, components_max_x, components_min_y, components_max_y


print("Create list of 4-connected components")
components, components_min_x, components_max_x, components_min_y, components_max_y = get_components(mask)


def put_red(components, input_image):
    image = np.copy(input_image)

    for i in range(0, len(components)):
        for j in range(0, len(components[i])):
            image[components[i][j][0]][components[i][j][1]][0] = 255
            image[components[i][j][0]][components[i][j][1]][1] = 0
            image[components[i][j][0]][components[i][j][1]][2] = 0

    return image


# TODO : there is clearly a problem with the min/max x/y rectangle with close component and close to borders,
# need to fix this.
def clean_image(components, components_min_x, components_max_x, components_min_y, components_max_y, input_image):
    image = np.copy(input_image)

    for i in range(0, len(components)):
        dist_x = components_max_x[i] - components_min_x[i]
        dist_y = components_max_y[i] - components_min_y[i]
        mimi = (image[components_min_x[i]-1][components_min_y[i]] + image[components_min_x[i]][components_min_y[i]-1])/2
        mima = (image[components_min_x[i]-1][components_max_y[i]] + image[components_min_x[i]][components_max_y[i]+1])/2
        mami = (image[components_max_x[i]+1][components_min_y[i]] + image[components_max_x[i]][components_min_y[i]-1])/2
        mama = (image[components_max_x[i]+1][components_min_y[i]] + image[components_max_x[i]][components_max_y[i]+1])/2

        for j in range(0, len(components[i])):
            x = components[i][j][0]
            y = components[i][j][1]

            prop1 = 1 - ((x-components_max_x[i]) / dist_x)
            prop2 = 1 - prop1
            prop3 = 1 - ((y-components_max_y[i]) / dist_y)
            prop4 = 1 - prop1

            col = (mimi * prop1 + mami * prop2 + mima * prop3 + mama * prop4) / 2

            image[x][y] = col

    return image


print("Remove every pixel of the mask in input image using a different rectangle of gradient")
cpt = 0
for file in file_names_list:
    print(file)
    # Image to view the components in red
    imageio.imwrite(os.path.join("./masks_" + str(user_threshold) + "_" + str(user_dilatation), file), put_red(components, images_list[cpt]))
    # Testing the solution on the first image
    imageio.imwrite(os.path.join("./results_" + str(user_threshold) + "_" + str(user_dilatation), file), clean_image(components, components_min_x, components_max_x, components_min_y, components_max_y, images_list[cpt]))
    cpt += 1

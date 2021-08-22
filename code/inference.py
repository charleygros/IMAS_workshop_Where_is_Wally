import numpy as np
from scipy.misc import imresize


def resize_image(img, size_patch):
    """Resizes image as a multiple of size_patch"""
    h, w, _ = img.shape
    n_vertical_patch = h / size_patch
    n_horizontal_patch = w / size_patch
    new_h, new_w = h, w
    if n_vertical_patch * size_patch != h:
        new_h = (n_vertical_patch + 1) * size_patch
    if n_horizontal_patch * size_patch != w:
        new_w = (n_horizontal_patch + 1) * size_patch
    if new_h == h and new_w == w:
        return img
    else:
        return imresize(img, (new_h, new_w))


def patchify_test_image(img, size_patch):
    """Extracts patches from testing image."""
    h, w, _ = img.shape
    n_vertical_patch = h / size_patch
    n_horizontal_patch = w / size_patch
    list_patch = []
    for i in range(n_vertical_patch):
        for j in range(n_horizontal_patch):
            list_patch.append(img[i * size_patch : (i + 1) * size_patch, j * size_patch : (j + 1) * size_patch])
    return np.stack(list_patch)


def reconstruct_image_from_patches(img, np_patches, size_patch):
    """Reconstruct image from stack of patches N x size_patch x size_patch."""
    h, w, _ = img.shape
    n_vertical_patch = h/size_patch
    n_horizontal_patch = w/size_patch
    img_reconstructed = []
    p = 0
    for i in range(n_vertical_patch):
        row = []
        for j in range(n_horizontal_patch):
            row.append(np_patches[p])
            p += 1
        img_reconstructed.append(np.concatenate(row, axis=1))
    return np.concatenate(img_reconstructed, axis=0)


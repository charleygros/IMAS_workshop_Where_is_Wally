import numpy as np


def patchify_test_image(img, size_patch):
    """Extracts patches from testing image."""
    h, w, _ = img.shape
    n_vertical_patch = int(h / size_patch)
    n_horizontal_patch = int(w / size_patch)
    list_patch = []
    for i in range(n_vertical_patch):
        for j in range(n_horizontal_patch):
            list_patch.append(img[i * size_patch : (i + 1) * size_patch, j * size_patch : (j + 1) * size_patch])
    return np.stack(list_patch)


def reconstruct_image_from_patches(img, np_patches, size_patch):
    """Reconstruct image from stack of patches N x size_patch x size_patch."""
    h, w, _ = img.shape
    n_vertical_patch = int(h / size_patch)
    n_horizontal_patch = int(w / size_patch)
    img_reconstructed = []
    p = 0
    for i in range(n_vertical_patch):
        row = []
        for j in range(n_horizontal_patch):
            row.append(np_patches[p])
            p += 1
        img_reconstructed.append(np.concatenate(row, axis=1))
    return np.concatenate(img_reconstructed, axis=0)


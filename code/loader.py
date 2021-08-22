import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

import utils as waldo_utils


class WaldoLoader(Dataset):
    """Waldo Loader."""

    def __init__(self, list_path_img, list_path_gt, size_img, size_patch, balance_positive=False, sequence_transforms=None):

        self.list_path_img = list_path_img
        self.list_path_gt = list_path_gt
        self.size_img = size_img
        self.size_patch = size_patch
        self.balance_positive = balance_positive
        self.sequence_transforms = sequence_transforms

        print("Resizing all images to same dimensions: {} pixels^2 ...".format(self.size_img))
        self.stack_img = np.stack([load_image(path_img, as_grayscale=False, size_img=self.size_img)
                                   for path_img in self.list_path_img])
        self.stack_gt = np.stack([load_image(path_gt, as_grayscale=True, size_img=self.size_img)
                                  for path_gt in self.list_path_gt])
        #self.stack_img = [load_image(path_img, as_grayscale=False, size_img=self.size_img)
        #                           for path_img in self.list_path_img]
        #self.stack_gt = [load_image(path_gt, as_grayscale=True, size_img=self.size_img)
        #                          for path_gt in self.list_path_gt]
        # Binary values for GTs
        #self.stack_gt = [g / 255 for g in self.stack_gt]
        self.stack_gt = self.stack_gt / 255
        assert(np.array_equal(self.stack_gt, self.stack_gt.astype(bool)))

        # Patch extraction
        self.list_patch_img, self.list_patch_gt = extract_all_patches(self.stack_img, self.stack_gt, self.size_patch)

        # Get Positive samples
        if self.balance_positive:
            self.list_patch_positive_img = []
            self.list_patch_positive_gt = []
            for i, gt in enumerate(self.stack_gt):
                coords_bbox = waldo_utils.find_bounding_box_coords(gt)
                self.list_patch_positive_img.append(extract_positive_patch(self.stack_img[i], coords_bbox, size_patch))
                self.list_patch_positive_gt.append(extract_positive_patch(gt, coords_bbox, size_patch))
        else:
            self.list_patch_positive_img, self.list_patch_positive_gt = None, None

    def __len__(self):
        return len(self.list_patch_img)

    def __getitem__(self, idx):
        if self.balance_positive and np.random.random() < 0.5:
            idx_ = int(idx * len(self.list_patch_positive_img) * 1. / len(self.list_patch_img))
            X = self.list_patch_positive_img[idx_]
            Y = self.list_patch_positive_gt[idx_]
        else:
            idx_ = None
            X = self.list_patch_img[idx]
            Y = self.list_patch_gt[idx]

        if self.sequence_transforms is not None:
            X, Y = self.sequence_transforms(X, Y)

        X = waldo_utils.StandardizeInstance()(X)
        print(X.mean(), Y.std())

        X = np.rollaxis(X, 2)

        assert (X.shape[1:] == Y.shape)
        assert (Y.shape == (self.size_patch, self.size_patch))
        assert(np.array_equal(Y, Y.astype(bool)))

        Y = np.expand_dims(Y, axis=0)
        print(X.shape, Y.shape)
        X = torch.from_numpy(X).float()
        #Y = np.array(waldo_utils.find_bounding_box_coords(Y), dtype=np.float32)
        Y = torch.from_numpy(Y)

        return X, Y


def extract_positive_patch(img, coords_bbox, size_patch):
    """Extracts patches around Bbox of a given patch size."""
    w_start, w_end, h_start, h_end = coords_bbox
    half_size_patch = size_patch // 2
    i = img[w_start-half_size_patch:w_start+half_size_patch,
        h_start-half_size_patch:h_start+half_size_patch]
    if i.shape[0] != size_patch or i.shape[1] != size_patch:
        i = img[w_end - size_patch:w_end,
               h_end - size_patch:h_end]
        if i.shape[0] != size_patch or i.shape[1] != size_patch:
            i = img[w_start:w_start+size_patch, h_start:h_start+size_patch]
        if i.shape[0] < size_patch or i.shape[1] < size_patch:
            i = np.zeros((size_patch, size_patch))
            i[w_start:w_start+size_patch, h_start:h_start+size_patch] = img[w_start:w_start+size_patch,
                                                                        h_start:h_start+size_patch]
        return i
    else:
        return i


def patchify(im, patch_size):
    """Extract non-overlapping patches from image."""
    patches = []
    width, height = im.shape[0], im.shape[1]
    is_2D = len(im.shape) == 2

    # If patch_size is not multiple of the image size, image is mirrored.
    if ((im.shape[0] % patch_size) != 0) or ((im.shape[1] % patch_size) != 0):
        im = mirror(im, patch_size)

    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            if is_2D:
                patch = im[i:i+patch_size, j:j+patch_size]
            else:
                patch = im[i:i+patch_size, j:j+patch_size, :]
            if patch.shape[:2] == (patch_size, patch_size):
                patches.append(patch)
    return patches


def mirror(im, length):
    """Mirrors an image on the right on length pixels."""
    width, height = im.shape[0], im.shape[1]
    is_2D = len(im.shape) == 2

    if is_2D:
        right_flipped = np.fliplr(im[width - length:, :])
    else:
        right_flipped = np.fliplr(im[width - length:, :, :])

    right_mirrored = np.concatenate((im, right_flipped), axis=0)

    if is_2D:
        bottom_flipped = np.flipud(right_mirrored[:, height - length:])
    else:
        bottom_flipped = np.flipud(right_mirrored[:, height - length:, :])

    mirrored = np.concatenate((right_mirrored, bottom_flipped), axis=1)
    return mirrored


def extract_all_patches(images, labels, patch_size):
    """Extracts patches for images and labels via patchify function."""

    # Get patches
    image_patches = [patch for im in images for patch in patchify(im, patch_size)]
    label_patches = [patch for label in labels for patch in patchify(label, patch_size)]
    assert(len(image_patches) == len(label_patches))

    return image_patches, label_patches


def load_image(path_img, as_grayscale=False, size_img=None):
    """Loads image and resizes it if img_size is not None."""
    img = Image.open(path_img)
    if as_grayscale:
        img = img.convert("L")
    if size_img:
        img = img.resize(size_img, Image.NEAREST)
    return np.array(img)

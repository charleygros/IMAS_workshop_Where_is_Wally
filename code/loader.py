import numpy as np
from PIL import Image
from skimage.transform import resize
from skimage.measure import label

import torch
from torch.utils.data import Dataset

import utils as waldo_utils


class WaldoLoader(Dataset):
    """Waldo Loader."""

    def __init__(self, list_path_img, list_path_gt, size_patch, balance_positive=False, sequence_transforms=None):

        self.list_path_img = list_path_img
        self.list_path_gt = list_path_gt
        self.size_patch = size_patch
        self.balance_positive = balance_positive
        self.sequence_transforms = sequence_transforms

        print("Loading images ...")
        self.list_img = [load_image(path_img, as_grayscale=False) for path_img in self.list_path_img]
        self.list_gt = [load_image(path_gt, as_grayscale=True) for path_gt in self.list_path_gt]
        print("Resizing image dimensions so that they are a multiple of the patch size: {} x {} pixels^2 ...".format(self.size_patch, self.size_patch))
        self.list_img = [resize_image(img, size_patch=self.size_patch) for img in self.list_img]
        self.list_gt = [gt / 255 for gt in self.list_gt]
        self.list_gt = [resize_image(gt, size_patch=self.size_patch) for gt in self.list_gt]
        self.list_gt = [waldo_utils.threshold_mask(gt, threshold=0.5) for gt in self.list_gt]

        # Patch extraction
        self.list_patch_img, self.list_patch_gt = extract_all_patches(self.list_img, self.list_gt, self.size_patch)

        # Get Positive samples
        if self.balance_positive:
            self.list_patch_positive_img = []
            self.list_patch_positive_gt = []
            for i, gt in enumerate(self.list_gt):
                assert (np.array_equal(gt, gt.astype(bool)))
                # In case there are multiple waldo inside an image
                np_gt_labeled, n_waldo = label(gt, return_num=True)
                for i_waldo in range(n_waldo):
                    gt_cur = (np_gt_labeled == i_waldo + 1).astype(int)
                    coords_bbox = waldo_utils.find_bounding_box_coords(gt_cur)
                    self.list_patch_positive_img.append(extract_positive_patch(self.list_img[i], coords_bbox, size_patch))
                    self.list_patch_positive_gt.append(extract_positive_patch(gt_cur, coords_bbox, size_patch))
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

        X = np.rollaxis(X, 2)

        assert (X.shape[1:] == Y.shape)
        assert (Y.shape == (self.size_patch, self.size_patch))
        assert(np.array_equal(Y, Y.astype(bool)))

        Y = np.expand_dims(Y, axis=0)
        X = torch.from_numpy(X.copy()).float()
        #Y = np.array(waldo_utils.find_bounding_box_coords(Y), dtype=np.float32)
        Y = torch.from_numpy(Y.copy())

        return X, Y


def extract_positive_patch(img, coords_bbox, size_patch):
    print(np.unique(img))
    """Extracts patches around Bbox of a given patch size."""
    w_start, w_end, h_start, h_end = coords_bbox
    w_middle = w_start + (w_end - w_start) // 2
    h_middle = h_start + (h_end - h_start) // 2
    half_size_patch = size_patch // 2
    i = img[w_middle-half_size_patch:w_middle+half_size_patch,
        h_middle-half_size_patch:h_middle+half_size_patch]
    #i = img[wend - 224:wstart + 224, hend - 224:hstart + 224]
    if i.shape[0] != size_patch or i.shape[1] != size_patch:
        print(i.shape, img.shape, w_middle, h_middle, w_start, w_end, h_start, h_end)
        i = img[w_end-size_patch:w_end, h_end-size_patch:h_end]
        if i.shape[0] != size_patch or i.shape[1] != size_patch:
            i = img[w_start:w_start + size_patch, h_start:h_start + size_patch]
        print("hey", i.shape)
        print(np.unique(img))
        return i
    else:
        return i


def patchify(im, patch_size):
    """Extract non-overlapping patches from image."""
    patches = []
    width, height = im.shape[0], im.shape[1]
    is_2D = len(im.shape) == 2

    # If patch_size is not multiple of the image size, image is mirrored.
    #if ((im.shape[0] % patch_size) != 0) or ((im.shape[1] % patch_size) != 0):
    #    im = mirror(im, patch_size)

    for i in range(0, width, patch_size):
        for j in range(0, height, patch_size):
            if is_2D:
                patch = im[i:i+patch_size, j:j+patch_size]
            else:
                patch = im[i:i+patch_size, j:j+patch_size, :]
            if patch.shape[:2] == (patch_size, patch_size):
                patches.append(patch)
    return patches


def extract_all_patches(images, labels, patch_size):
    """Extracts patches for images and labels via patchify function."""

    # Get patches
    image_patches = [patch for im in images for patch in patchify(im, patch_size)]
    label_patches = [patch for label in labels for patch in patchify(label, patch_size)]
    assert(len(image_patches) == len(label_patches))

    return image_patches, label_patches


def load_image(path_img, as_grayscale=False):
    """Loads image and resizes it if img_size is not None."""
    img = Image.open(path_img)
    if as_grayscale:
        img = img.convert("L")
    return np.array(img)


def resize_image(img, size_patch):
    """Resizes image as a multiple of size_patch"""
    h, w = img.shape[:2]
    if h % size_patch != 0:
        new_h = (int(h / size_patch) + 1) * size_patch
    else:
        new_h = h
    if w % size_patch != 0:
        new_w = (int(w / size_patch) + 1) * size_patch
    else:
        new_w = w
    if new_h == h and new_w == w:
        return img
    else:
        return resize(img, (new_h, new_w))

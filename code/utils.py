import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def get_fname_list(root, extension):
    """Returns a list of filenames under root directory with a given extension."""
    return [os.path.join(directory_path, f) for directory_path, directory_name,
            files in os.walk(root) for f in files if f.endswith(extension)]


def get_annotation_info(path):
    """Returns pandas dataframe containing the annotation infos."""
    return pd.read_csv(path)


def create_binary_mask(list_bbox_coords, im_ref_width, im_ref_height):
    """Creates a mask for the bounding box of same shape as image."""
    binary_mask = np.zeros((im_ref_height, im_ref_width), dtype=np.float16)
    for bbox_coords in list_bbox_coords:
        bbox_coords = np.array(bbox_coords).astype(np.int)
        binary_mask[bbox_coords[1]:bbox_coords[3], bbox_coords[0]:bbox_coords[2]] = 1.
    return binary_mask


def find_bounding_box_coords(gt):
    """Finds bounding box coordinates."""
    shape_ = gt.shape
    # Size of the bbox
    w = np.max(np.unique(gt.sum(axis=0)))
    h = np.max(np.unique(gt.sum(axis=1)))
    # Start of bbox
    w_start = np.argmax(gt.sum(axis=1))
    h_start = np.argmax(gt.sum(axis=0))
    # End of bbox
    w_end = int(w_start + w)
    h_end = int(h_start+h)
    return (w_start, w_end, h_start, h_end)


def imshow_tensor(img, fname=None):
    """
    Display an image given as tensor.
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if fname:
        plt.savefig(fname)
    plt.show()


def imshow_tensor_gt(img, fname=None):
    """
    Display label image given as tensor.
    """
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if fname:
        plt.savefig(fname)
    plt.show()


class StandardizeInstance(object):
    """Normalize a tensor or an array image with mean and standard deviation estimated from the sample itself."""
    def __call__(self, sample):
        data_out = (sample - sample.mean()) / sample.std()
        return data_out

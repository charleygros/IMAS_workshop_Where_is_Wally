import random
import numpy as np


class Sequence(object):
    """Initialise Sequence object for transformations"""

    def __init__(self, augmentations, probs=1):

        self.augmentations = augmentations
        self.probs = probs

    def __call__(self, images, bboxes):
        for i, augmentation in enumerate(self.augmentations):
            if type(self.probs) == list:
                prob = self.probs[i]
            else:
                prob = self.probs

            if np.random.random() < prob:
                images, bboxes = augmentation(images, bboxes)

        return images, bboxes


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the Image with the probability p."""

    def __call__(self, img, bbox):
        img_ = np.fliplr(img)
        bbox_ = np.fliplr(bbox)

        return img_, bbox_


class RandomTranslate(object):
    """Randomly Translates the image.

    Note: If the resulting bbox has less than 25% of the initial coverage, then returns empty array.
    """

    def __init__(self, translate=0.2, diff=False):
        self.translate = translate

        if type(self.translate) == tuple:
            assert len(self.translate) == 2, "Invalid range"
            assert self.translate[0] > 0 & self.translate[0] < 1
            assert self.translate[1] > 0 & self.translate[1] < 1
        else:
            assert self.translate > 0 and self.translate < 1
            self.translate = (-self.translate, self.translate)

        self.diff = diff

    def __call__(self, img, bbox):
        # Chose a random digit to scale by
        img_shape = img.shape

        # translate the image

        # percentage of the dimension of the image to translate
        translate_factor_x = random.uniform(*self.translate)
        translate_factor_y = random.uniform(*self.translate)

        if not self.diff:
            translate_factor_y = translate_factor_x
        corner_x = int(translate_factor_x * img.shape[1])
        corner_y = int(translate_factor_y * img.shape[0])
        # change the origin to the top-left corner of the translated box
        orig_box_cords = [max(0, corner_y), max(corner_x, 0), min(img_shape[0], corner_y + img.shape[0]),
                          min(img_shape[1], corner_x + img.shape[1])]

        canvas = np.zeros(img_shape).astype(np.uint8)
        print(np.unique(img),"1")
        mask = img[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]),
               max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1]), :]
        canvas[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3], :] = mask
        img = canvas
        print(np.unique(img), "2")
        bbox_area_before = np.sum(bbox)
        canvas_bbox = np.zeros(bbox.shape).astype(np.uint8)
        mask_bbox = bbox[max(-corner_y, 0):min(img.shape[0], -corner_y + img_shape[0]),
               max(-corner_x, 0):min(img.shape[1], -corner_x + img_shape[1])]
        canvas_bbox[orig_box_cords[0]:orig_box_cords[2], orig_box_cords[1]:orig_box_cords[3]] = mask_bbox
        bbox = canvas_bbox
        bbox_area_after = np.sum(bbox)

        if bbox_area_after < 0.25 * bbox_area_before:
            bbox = np.zeros(bbox.shape)
        print(np.unique(img))
        return img, bbox

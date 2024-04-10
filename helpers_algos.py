import numpy as np
import cv2

mask_types = {
    'sure_fg': [255, 255, 255],
    'sure_bg': [0, 0, 0],
    'probable_fg': [200, 200, 200],
    'probable_bg': [100, 100, 100],
}


def grab_cut(img, mask):

    mask[mask == 0] = cv2.GC_BGD
    mask[mask == 100] = cv2.GC_PR_BGD
    mask[mask == 200] = cv2.GC_PR_FGD
    mask[mask == 255] = cv2.GC_FGD

    mask = np.reshape(mask, (mask.shape[0], mask.shape[1],))

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    img, bgdModel, fgdModel = cv2.grabCut(img, mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)

    return np.where((mask == 2) | (mask == 0), 0, 255).astype('uint8')


def draw_superpixels(image, superpixels):
    mask = superpixels.getLabelContourMask()

    image_with_boundaries = image.copy()
    image_with_boundaries[mask == 255] = [0, 0, 255]

    return image_with_boundaries


def flood_fill(image, seed_point, fill_color=(0, 255, 0), diff=(3, 3, 3)):
    # Create a mask to keep track of filled pixels
    mask = np.zeros((image.shape[0] + 2, image.shape[1] + 2), dtype=np.uint8)

    # Flood fill
    cv2.floodFill(image, mask, seedPoint=seed_point, newVal=fill_color, loDiff=diff, upDiff=diff)

    return mask[1:-1, 1:-1]


def superpixels_to_select(image, seed_point, superpixels, diff):
    fill_color = (0, 255, 0)  # Green

    image_flood_filled = image.copy()
    mask = flood_fill(image_flood_filled, seed_point, fill_color, diff)

    labels = superpixels.getLabels()

    return np.unique(labels[mask == 1])


def draw_mask(current_mask, labels, superpixels_with_fill, action):
    for i in superpixels_with_fill:
        mask = labels == i
        current_mask[mask] = mask_types[action]
    return current_mask

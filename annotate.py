import os

from helpers_algos import *

# Create a dictionary to store settings
settings = {
    'action': None,
    'autoexpand': False,
    'diff': 3,
    'curr_image': 0,
    'scale': 2
}


def load_images(dataset_path='./JPEGImages/', label_path='./SegmentationObject/'):
    files = os.listdir(dataset_path)
    data = []

    for file_name in files:
        img = cv2.imread(dataset_path + file_name)
        label = cv2.imread(label_path + file_name[:-4] + '.png')

        lsc_sup = cv2.ximgproc.createSuperpixelLSC(cv2.cvtColor(img, cv2.COLOR_BGR2LAB), region_size=10)
        lsc_sup.iterate(10)

        img_with_boundaries = draw_superpixels(img, lsc_sup)
        empty_mask = np.zeros(img.shape, np.uint8)

        width = int(img.shape[1] * 2)
        height = int(img.shape[0] * 2)
        dim = (width, height)

        data.append({'image': img,
                     'image_with_boundaries': cv2.resize(img_with_boundaries, dim),
                     'lsc': lsc_sup,
                     'mask': empty_mask,
                     'label': label})
    return data


def get_curr_image():
    global image
    global image_with_boundaries
    global mask
    global lsc

    image = images[settings['curr_image']]['image']
    image_with_boundaries = images[settings['curr_image']]['image_with_boundaries']
    mask = images[settings['curr_image']]['mask']
    lsc = images[settings['curr_image']]['lsc']
    settings['curr_image'] += 1


def compute_dice(final_mask, label):
    label_bool = (label[:, :, 0] == 0) & (label[:, :, 1] == 0) & (label[:, :, 2] == 128)

    final_mask_bool = final_mask.astype(bool)

    return 2 * np.logical_and(label_bool, final_mask_bool).sum() / (
                label_bool.sum() + final_mask_bool.sum())


def mouse_callback(event, x, y, flags, param):
    global image
    global mask

    if event == cv2.EVENT_LBUTTONDOWN:
        to_select = superpixels_to_select(image, (int(x/2), int(y/2)), lsc, (settings['diff'], settings['diff'], settings['diff']))
        mask = draw_mask(mask, lsc.getLabels(), to_select, settings['action'])


images = load_images()

image = None
image_with_boundaries = None
mask = None
lsc = None

get_curr_image()

cv2.namedWindow('Image')
cv2.namedWindow('Mask')
cv2.namedWindow('GrabCut Mask')

cv2.setMouseCallback('Image', mouse_callback)
while True:
    if 255 in mask:
        gc_mask = grab_cut(image.copy(), cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY))
    else:
        gc_mask = mask

    cv2.imshow('Image', image_with_boundaries)
    cv2.imshow('Mask', mask)
    cv2.imshow('GrabCut Mask', gc_mask)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('1'):
        settings['action'] = 'sure_fg'
        print('selecting sure foreground')
    elif key == ord('2'):
        settings['action'] = 'probable_fg'
        print('selecting probable foreground')
    elif key == ord('3'):
        settings['action'] = 'probable_bg'
        print('selecting probable background')
    elif key == ord('4'):
        settings['action'] = 'sure_bg'
        print('selecting sure background')

    elif key == ord('+'):
        settings['diff'] += 1
        print(f"current diff: {settings['diff']}")
    elif key == ord('-'):
        settings['diff'] -= 1 if settings['diff'] > 0 else 0
        print(f"current diff: {settings['diff']}")
    elif key == ord('c'):
        s_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, s_element)
    elif key == ord('d'):
        print(f"dice score: {compute_dice(gc_mask, images[settings['curr_image']-1]['label'])}")
    elif key == ord('n'):
        if settings['curr_image'] >= len(images):
            break
        get_curr_image()

cv2.destroyAllWindows()

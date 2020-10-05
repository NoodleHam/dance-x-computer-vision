import glob
from os import path, mkdir
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np
import tqdm

from utils.color_temp import convert_temperature

METADATA_DIR = "hands_data"
VERSION_NAME = "v4"
DATASET_NAME = "hands_right"
SAVE_DIR = path.join(METADATA_DIR, DATASET_NAME + "_" + VERSION_NAME)

save_file_counter = 0


def save_foreground_img(foreground_img, filename):
    """
    Saves the foreground image to disk with the background being transparent
    :param foreground_img: the image of the hand in that has been cropped, with background pixels strictly black
    """
    background_mask = np.max(foreground_img[..., :], 2) == 0
    foreground_img = cv2.cvtColor(foreground_img, cv2.COLOR_RGB2BGRA)
    foreground_img[background_mask, 3] = 0
    cv2.imwrite(filename, foreground_img)


def autocrop(image, vertical_threshold=60, horizontal_threshold=30):
    """Crops any edges below or equal to threshold
    Returns cropped image.
    """
    if len(image.shape) == 3:
        flatImage = np.max(image, 2)
    else:
        flatImage = image
    assert len(flatImage.shape) == 2

    crop_upper_left = (0,0)
    nonblack_row_indices = np.where(np.max(flatImage, 1) > vertical_threshold)[0]
    if nonblack_row_indices.size:
        # eliminate useless rows first
        image = image[nonblack_row_indices[0]: nonblack_row_indices[-1] + 1, :]
        # then crop columns
        nonblack_col_indices = np.where(np.max(image, 0) > horizontal_threshold)[0]
        image = image[:, nonblack_col_indices[0]: nonblack_col_indices[-1] + 1]
        crop_upper_left = (nonblack_col_indices[0], nonblack_row_indices[0])
    else:
        image = image[:1, :1]
    return image, crop_upper_left


def max_area_contour(contour_list):
    if len(contour_list) == 0:
        return None
    max_i = 0
    max_area = 0

    for i in range(len(contour_list)):
        cnt = contour_list[i]

        area_cnt = cv2.contourArea(cnt)

        if area_cnt > max_area:
            max_area = area_cnt
            max_i = i

    return contour_list[max_i]


def get_contours(binary_img):
    # gray_hist_mask_image = cv2.cvtColor(hist_mask_image, cv2.COLOR_BGR2GRAY)
    # ret, thresh = cv2.threshold(gray_hist_mask_image, 0, 255, 0)
    cont, hierarchy = cv2.findContours(np.uint8(binary_img), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cont


def contour_effects(binary_mask):
    mask = binary_mask.copy()
    contour = max_area_contour(get_contours(mask))
    mask_contour = cv2.cvtColor(np.uint8(mask*255), cv2.COLOR_GRAY2BGR)
    # mask_contour = cv2.drawContours(cv2.cvtColor((mask) * np.uint8(255), cv2.COLOR_GRAY2BGR), [contour], 0,
    #                                 (0, 255, 255), 5)
    if mask_contour is not None:
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        # far_points = farthest_points(defects, max_cont, cnt_centroid)
        contour_hull_points = []
        contour_defect_points = []
        for hull_index in hull:
            contour_hull_points.append(contour[hull_index[0]][0])
            # cv2.circle(mask_contour, tuple(contour[hull_index][0][0]), 5, [0, 0, 255], -1)
        contour_hull_points = np.array(contour_hull_points)
        contour_hull_points = contour_hull_points.reshape((-1, 1, 2))
        for defect in defects:
            s, e, f, d = defect[0]
            contour_defect_points.append(contour[f][0])
        contour_defect_points = np.array(contour_defect_points, np.int32)
        # print(contour_defect_points)
        # contour_defect_points = contour_defect_points.reshape((-1, 1, 2))
        mask_contour = cv2.polylines(mask_contour, [contour_hull_points],
                              True, [255, 0, 0], 3)
        mask_contour = cv2.polylines(mask_contour, [contour_defect_points],
                                     True, [128, 0, 0], 3)
        for hull_index in hull:
            cv2.circle(mask_contour, tuple(contour[hull_index][0][0]), 5, [0, 0, 255], -1)
        for i in range(len(defects)):
            cv2.circle(mask_contour, tuple(list(contour_defect_points[i])), 5, [200, 0, 0], -1)

    return mask_contour



def remove_background(image,):
    image = convert_temperature(image, temp=7000)
    image_copy = np.copy(image)
    # convert original to RGB
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    # if(save_file_counter > 100):
    #     plt.imshow(image_copy)
    #     plt.show()
    # reserve true blacks as for mask
    # create mask
    # mask_background = cv2.inRange(image_copy, lower_clr_bound, upper_clr_bound)
    r_channel = image_copy[..., 0].astype(int)
    g_channel = image_copy[..., 1].astype(int)
    b_channel = image_copy[..., 2].astype(int)
    rb_minus_2g = (r_channel + b_channel - g_channel * 2)
    # plt.imshow(r_channel)
    # plt.show()
    # plt.imshow(b_channel)
    # plt.show()
    # plt.imshow(rb_difference, cmap='gray')
    # plt.show()
    # use mask
    masked_image = np.copy(image_copy)
    # background: r+g - 2b is very negative and b channel is bright
    # masked_image[(rg_minus_2b < -19*2) | (b_channel > 190)] = [0, 0, 0]
    mask = rb_minus_2g < -39*2
    masked_image[mask] = [0, 0, 0]
    contour_efx_img = contour_effects(~mask)
    cv2.imshow('mask', contour_efx_img)
    cv2.imshow('masked_image', cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)
    # mask_edges = masked_image[..., 1] > masked_image[..., 0]
    # masked_image[mask_edges] = [0, 0, 0]
    cropped_image, crop_upper_left = autocrop(masked_image)
    # if (save_file_counter > 100):
    #     plt.imshow(cropped_image)
    #     plt.show()
    return cropped_image, crop_upper_left

def get_filename(save_dir, file_counter):
    return path.join(save_dir, DATASET_NAME + "_" + str(file_counter) + ".png")


def main():
    video_files = glob.glob('videos/*.mp4')
    video_files.extend(glob.glob('videos/*.m4v'))
    cap = cv2.VideoCapture(video_files[0])
    # lower_green = np.array([0, 120, 0])     ##[R value, G value, B value]
    # upper_green = np.array([100, 255, 100])
    _, original_img = cap.read()

    image_metadata_list = []
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Found " + str(length) + " frames in input video")
    pbar = tqdm.tqdm(total=length)
    # quit()
    global save_file_counter
    for i in range(20):
        _, __ = cap.read()
    try:
        while cap.isOpened():
            _, original_img = cap.read()
            if len(original_img) < 48 or len(original_img[0]) < 48:
                break
            foreground, crop_ul = remove_background(original_img)
            # ignore potential hands with size smaller than 48*48
            if len(foreground) < 48 or len(foreground[0]) < 48:
                continue
            save_file_counter += 1
            pbar.update(1)
                #print(json.dumps(entry))
            # plt.imshow(original_img)
            # plt.show()

    except (KeyboardInterrupt, SystemExit):
        print("Aborting ground truth generation. Saving partial metadata...")
    except None: #TypeError
        print("Video frames ended prematurely. Saving partial metadata...")
    #process unkown err

    # with open(SAVE_DIR + '_metadata.json', 'w') as outfile:
    #     for entry in image_metadata_list:
    #         json.dump(entry, outfile)
    #         outfile.write('\n')


if __name__ == "__main__":
    main()
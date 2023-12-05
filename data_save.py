import os
from pathlib import Path

import cv2 as cv
import numpy as np
from common.data_parsing import write_toml
from common.fitting import fit_min_area_rect


def save_data(simple_path_prefix, simple_segm_img, segments_info, scale_to_or, segm_img=None):

    base = os.path.split(simple_path_prefix)[0]
    Path(base).mkdir(parents=True, exist_ok=True)

    for i in range(1, len(segments_info) + 1):
        pixels_to_fit = np.array(np.where(simple_segm_img == i)).T
        box = fit_min_area_rect(pixels_to_fit)
        box *= scale_to_or
        segments_info[i - 1]["box"] = box.tolist()
    segments_info = {"objects": segments_info}

    if segm_img is not None:
        cv.imwrite(f"{simple_path_prefix}_segmentation_original.png", segm_img)

    cv.imwrite(f"{simple_path_prefix}_segmentation.png", simple_segm_img)
    # img_read = cv.imread(f"{simple_path_prefix}_segmentation.png")
    # assert img_read == simple_segm_img

    write_toml(segments_info, f"{simple_path_prefix}_data.toml")
    # data_read = read_toml(f"{simple_path_prefix}_data.toml")
    # assert segments_info == data_read
    return segments_info

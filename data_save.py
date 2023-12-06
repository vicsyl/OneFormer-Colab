import os
from pathlib import Path

import cv2 as cv

from common.data_parsing import write_toml


def save_data(simple_path_prefix, simple_segm_img, segments_info, scale_to_or, segm_img=None):

    base = os.path.split(simple_path_prefix)[0]
    Path(base).mkdir(parents=True, exist_ok=True)

    if segm_img is not None:
        cv.imwrite(f"{simple_path_prefix}_segmentation_original.png", segm_img)

    cv.imwrite(f"{simple_path_prefix}_segmentation.png", simple_segm_img)
    # img_read = cv.imread(f"{simple_path_prefix}_segmentation.png")
    # assert img_read == simple_segm_img

    segments_info = {"objects": segments_info}
    write_toml(segments_info, f"{simple_path_prefix}_data.toml")
    # data_read = read_toml(f"{simple_path_prefix}_data.toml")
    # assert segments_info == data_read
    return segments_info

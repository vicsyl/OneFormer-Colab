import os
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
from copy import deepcopy

import tomli
import tomli_w
import cv2 as cv


# TOML #
def none_to_string(el):

    if type(el) in [dict, OrderedDict]:
        for k, v in el.copy().items():
            if k is None:
                el.remove(None)
                el["None"] = v
                k = "None"
            if v is None:
                el[k] = "None"
            else:
                none_to_string(v)

    elif type(el) == list:
        for idx, e in enumerate(el.copy()):
            if e is None:
                el[idx] = "None"
            else:
                none_to_string(e)


def string_to_none(el):

    if type(el) in [dict, OrderedDict]:
        for k, v in el.copy().items():
            if k == "None":
                el.remove("None")
                el[None] = v
                k = None
            if v == "None":
                el[k] = None
            else:
                string_to_none(v)

    elif type(el) == list:
        for idx, e in enumerate(el.copy()):
            if e == "None":
                el[idx] = None
            else:
                string_to_none(e)


def write_toml(mmap, file_path):
    mmap = deepcopy(mmap)
    none_to_string(mmap)
    max_retries = 0
    for retry in range(max_retries + 1):
        try:
            with open(file_path, "w") as fd:
                toml = tomli_w.dumps(mmap)
                fd.write(f'{toml}\n')
            break
        except OSError:
            if retry == max_retries:
                E, V, T = sys.exc_info()
                raise E(V).with_traceback(T)
            print(f"OSError caught, retry no. {retry + 1} ...")


def read_toml(file_path):
    with open(file_path, "rb") as f:
        toml_dict = tomli.load(f) # , parse_float=Decimal)
        string_to_none(toml_dict)
        return toml_dict


def fit_min_area_rect(pixels_to_fit):
    rect = cv.minAreaRect(pixels_to_fit) # (x, y), (width, height), angle_of_rotation
    box = cv.boxPoints(rect)
    return box


def save_data(simple_path_prefix, img_path, panoptic_seg, segments_info, scale_to_or, segm_img=None):

    base = os.path.split(simple_path_prefix)[0]
    Path(base).mkdir(parents=True, exist_ok=True)

    seg_np = panoptic_seg.cpu().numpy()
    for i in range(1, len(segments_info) + 1):
        pixels_to_fit = np.where(seg_np == i)
        box = fit_min_area_rect(pixels_to_fit)
        # box = np.array([[0, 0], [0, 1], [1, 1], [1, 0]]).astype(float)
        box *= scale_to_or
        segments_info[i - 1]["box"] = box.tolist()
    segments_info = {"objects": segments_info}

    # simple_segm_img = seg_np.copy().to(float) / 255
    simple_segm_img = seg_np

    if segm_img:
        cv.imwrite(f"{simple_path_prefix}_segmentation_original.png", segm_img)

    cv.imwrite(f"{simple_path_prefix}_segmentation.png", simple_segm_img)
    img_read = cv.imread(f"{simple_path_prefix}_segmentation.png")
    assert img_read == simple_segm_img

    write_toml(segments_info, f"{simple_path_prefix}_data.toml")

    data_read = read_toml(f"{simple_path_prefix}_data.toml")
    assert segments_info == data_read
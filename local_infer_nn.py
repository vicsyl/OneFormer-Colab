
import cv2
import numpy as np

from data_save import save_data
from infer import TASK_INFER, predictor, metadata
import imutils


def infer_and_save(img_path, img_name, out_data_root):

    task = "panoptic"

    # img_path = config_entry['orig_file_path']
    # hack - ../../download/... -> ./download/.
    # img_path = img_path[4:]

    # print(f"path: {img_path}")
    img_fpath = f"{img_path}/{img_name}"
    original_img = cv2.imread(img_fpath)
    or_size = original_img.shape[:2]

    resized_img = imutils.resize(original_img, width=640)
    size = resized_img.shape[:2]

    scale_to_or_0 = float(or_size[0]) / float(size[0])
    scale_to_or_1 = float(or_size[1]) / float(size[1])
    # print(f"scale_to_or_0: {scale_to_or_0}")
    # print(f"scale_to_or_1: {scale_to_or_1}")

    # MAY not hold actually
    assert np.isclose(scale_to_or_0, scale_to_or_1)

    predictions, out = TASK_INFER[task](resized_img, predictor, metadata)
    segm_vis_img = out.get_image()
    segmentation_map, segments_info = predictions["panoptic_seg"]

    simple_path_prefix = f"{out_data_root}/{img_name}"[:-4]
    print(f"simple_path_prefix: {simple_path_prefix}")

    segmentation_map = segmentation_map.cpu().numpy()
    segments_info = save_data(simple_path_prefix, segmentation_map, segments_info, scale_to_or_0, segm_vis_img)

    contrast = 230 // segmentation_map.max()
    segm_contrast_img = segmentation_map * contrast
    segm_contrast_img[segm_contrast_img == 0] = 255

    return segments_info, segmentation_map, segm_contrast_img, segm_vis_img, original_img


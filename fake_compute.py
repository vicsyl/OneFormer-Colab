import time

from sympy import Quaternion

from data_parsing import save, get_cached_data, ConfStatic, read_toml
from data_save import save_data
import cv2 as cv
import numpy as np
import argparse

from common.common_transforms import project_metropolis
from common.vanishing_point import change_x_3d_arkit, change_r_arkit


def scene_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_data_root",
        default="./out_data",
        help="output folder"
    )
    parser.add_argument("--conf_base_path", type=str, default=None)
    parser.add_argument("--cache_every_other", type=int, default=10000)
    parser.add_argument("--format_suffix", type=str, default=ConfStatic.toml_suffix)
    args = parser.parse_args()
    print(f"Args: {args}")
    return args


BOXES_2D_KEY = "segmented_boxes_2d"
WIDTHS_HEIGHTS_KEY = "segmented_wh"


def compute(fc=None):

    task = "panoptic"
    args = scene_args()

    # toml_conf = read_toml(args.conf_base_path)
    # configs/ARKitScenes=obj=2_max=100_sp=10.toml => conf_base_path = configs/ARKitScenes=obj=2_max=100

    start_time = time.time()
    data_entries, ready_entries, min_counts_map, config_read = get_cached_data(args.conf_base_path,
                                                                               format_suffix=ConfStatic.toml_suffix,
                                                                               out_log=True)

    for e_i, e in enumerate(data_entries):

        if e.__contains__(BOXES_2D_KEY):
            continue
        else:
            ready_entries += 1
            # if ready_entries > 10:
            #     break

        img_path = e['orig_file_path']
        # hack - ../../download/... -> ./download/.
        img_path = img_path[4:]
        print(f"path: {img_path}")

        assumed_prefix_l = len("./download/3dod/Training/")
        simple_path_prefix = f"{args.out_data_root}/{img_path[assumed_prefix_l:]}"[:-4]

        # START FUNCTION
        # segments_info, segmentation, scale = fc(img_path, simple_path_prefix)

        img = cv.imread(img_path)
        or_size = img.shape[:2]

        toml_path = f"{simple_path_prefix}_data.toml"
        segments_info = read_toml(toml_path)

        # segm_original_path = f"{simple_path_prefix}_segmentation_original.png"

        segmentation = f"{simple_path_prefix}_segmentation.png"
        # TODO type !!!
        img_segm = cv.imread(segmentation)
        size = img_segm.size[:2]

        scale = float(or_size[0]) / float(size[0])
        scale2 = float(or_size[1]) / float(size[1])
        # MAY not hold actually
        assert np.isclose(scale, scale2)

        # -> segments_info, segmentation, scale
        # ret = segments_info, img_segm, scale
        # END FUNCTION

        # MATCHING NO 1. -> visualize!!!
        K = e['K']
        R_cs_l = e['R_cs'].copy()
        R_gt = Quaternion(R_cs_l).rotation_matrix
        R_gt = change_r_arkit(R_gt)

        t_cs_l = e['t_cs'].copy()
        t_gt = np.array(t_cs_l)

        X_i = e['X_i']
        X_i = np.array(change_x_3d_arkit(np.array(X_i))).tolist()

        x_i_gt = project_metropolis(K, R_gt, t_gt, X_i)

        # CONTINUE: -> and check this!!
        # this is already easy to debug!!!
        x_i_gt = np.round(x_i_gt / scale)
        obj_idx = img_segm[x_i_gt]

        # this is probably better to have all bb's rather than computing them lazily
        bb = segments_info[obj_idx]
        # check the class
        e['names'][obj_idx] == bb['class'] #??
        # save_data(simple_path_prefix, panoptic_seg, segments_info, scale_to_or_0, segm_img)

        if ready_entries % args.cache_every_other == 0:
            sp_file_path = f"{args.conf_base_path}_sp={ready_entries}"
            save(f"{sp_file_path}{args.format_suffix}",
                 data_entries,
                 objects_counts_map={},
                 at_least_objects_counts_map=min_counts_map,
                 conf_attribute_map=config_read)

    print("Saving the final file")
    elapased = time.time() - start_time
    print(f"Total time: %f sec" % elapased)

    sp_file_path = f"{args.conf_base_path}_sp={ready_entries}"
    save(f"{sp_file_path}{args.format_suffix}",
         data_entries,
         objects_counts_map={},
         at_least_objects_counts_map=min_counts_map,
         conf_attribute_map=config_read)


if __name__ == "__main__":
    compute()

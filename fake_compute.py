import argparse
import time

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

from metadata import *
from common.common_transforms import project_metropolis
from common.data_parsing import save, get_cached_data, ConfStatic, read_toml
from common.vanishing_point import change_x_3d_arkit, change_r_arkit
from common.fitting import fit_min_area_rect


def scene_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_data_root",
        default="./out_data",
        help="output folder"
    )
    parser.add_argument("--max_entries", type=int, default=None)
    parser.add_argument("--conf_base_path", type=str, default=None)
    parser.add_argument("--cache_every_other", type=int, default=5000)
    parser.add_argument("--format_suffix", type=str, default=ConfStatic.toml_suffix)
    args = parser.parse_args()
    print(f"Args: {args}")
    return args


BOXES_2D_KEY = "segmented_boxes_2d_1"
WIDTHS_HEIGHTS_KEY = "segmented_wh"


def simple_show_img(img_l, title=None):
    plt.figure()
    plt.title(title)
    plt.imshow(img_l)
    plt.show()
    plt.close()


def get_scale(original_img, segmemtation_map):
    or_size = original_img.shape[:2]
    size = segmemtation_map.shape[:2]
    scale = float(or_size[0]) / float(size[0])
    scale2 = float(or_size[1]) / float(size[1])
    # MAY not hold actually
    assert np.isclose(scale, scale2)
    return scale


def get_simple_path_prefix(config_entry, out_data_root):

    img_path = config_entry['orig_file_path']
    # hack - ../../download/... -> ./download/.
    img_path = img_path[4:]
    print(f"path: {img_path}")
    assumed_prefix_l = len("./download/3dod/Training/")
    return f"{out_data_root}/{img_path[assumed_prefix_l:]}"[:-4]


def get_imgs(config_entry, out_data_root):

    img_path = config_entry['orig_file_path']

    # # hack - ../../download/... -> ./download/.
    img_path = img_path[4:]
    # print(f"path: {img_path}")
    # assumed_prefix_l = len("./download/3dod/Training/")
    # simple_path_prefix = f"{out_data_root}/{img_path[assumed_prefix_l:]}"[:-4]
    simple_path_prefix = get_simple_path_prefix(config_entry, out_data_root)

    original_img = cv.imread(img_path)
    # simple_show_img(original_img, "original image")

    segments_info = read_toml(f"{simple_path_prefix}_data.toml")
    segm_vis_img = cv.imread(f"{simple_path_prefix}_segmentation_original.png")
    # simple_show_img(segm_vis_img, "visualizer")

    segmentation_map = cv.imread(f"{simple_path_prefix}_segmentation.png").astype(int)[:, :, 0]
    contrast = 230 // segmentation_map.max()
    segm_contrast_img = segmentation_map * contrast
    segm_contrast_img[segm_contrast_img == 0] = 255
    # simple_show_img(segm_contrast_img, "segm_contrast_img")
    # simple_show_img(segmentation_map, "segmentation_map")
    return segments_info, segmentation_map, segm_contrast_img, segm_vis_img, original_img


if_thing_stuff_maps = {
    True: (thing_dataset_id_to_contiguous_id, thing_classes),
    False: (stuff_dataset_id_to_contiguous_id, stuff_classes),
}


def get_boxes(config_entry, segments_info, segmemtation_map, scale):

    K = np.array(config_entry['K'].copy())
    R_cs_l = config_entry['R_cs'].copy()
    R_gt = Quaternion(R_cs_l).rotation_matrix
    R_gt = change_r_arkit(R_gt)
    t_gt = np.array(config_entry['t_cs'].copy())
    X_i = np.array(config_entry['X_i'].copy())
    X_i = change_x_3d_arkit(X_i)
    X_i_test = np.array(config_entry["X_i_new"].copy())
    assert np.allclose(np.array(X_i_test), np.array(X_i))
    x_i_gt = project_metropolis(K, R_gt, t_gt, X_i)
    x_i_int_unscaled = np.round(x_i_gt / scale).astype(int)
    x_i_int = np.round(x_i_gt).astype(int)

    # even missing for these
    categories = []
    ds_categories = []
    boxes_original = []
    boxes_unscaled = []
    # split these
    x_i_int_unscaled_ret = []
    x_i_int_ret = []
    x_i_int_out_unscaled_ret = []
    x_i_int_out_ret = []
    for i, coord_x_y in enumerate(x_i_int_unscaled):

        def clean():
            boxes_unscaled.append([])
            boxes_original.append([])
            x_i_int_out_unscaled_ret.append(coord_x_y)
            x_i_int_out_ret.append(x_i_int[i])
            categories.append("not found")

        ds_name = config_entry['names'][i]
        ds_categories.append(ds_name)

        if coord_x_y[0] < 0 or coord_x_y[0] >= segmemtation_map.shape[1] or \
                coord_x_y[1] < 0 or coord_x_y[1] >= segmemtation_map.shape[0]:
            clean()
            continue
        obj_idx = segmemtation_map[coord_x_y[1], coord_x_y[0]]
        if obj_idx == 0:
            clean()
            continue

        x_i_int_unscaled_ret.append(coord_x_y)
        x_i_int_ret.append(x_i_int[i])

        # object_info = "{'id': 1,
        # 'isthing': False,
        # 'category_id': 0,
        # 'area': 112475.0,
        # 'box': list[Float, 4, 2]}"
        object_info = segments_info['objects'][obj_idx - 1]
        assert object_info["id"] == obj_idx

        # isthing = object_info["isthing"]
        # well it is just using stuff IMHO
        id_map, classes_map = if_thing_stuff_maps[False]
        id = id_map[object_info["category_id"]]
        assert id == object_info["category_id"]
        category = classes_map[id]
        categories.append(category)

        pixels_to_fit = np.array(np.where(segmemtation_map == obj_idx)).T
        box = fit_min_area_rect(pixels_to_fit)
        boxes_unscaled.append(box.tolist().copy())
        box *= scale
        box = box.tolist()
        boxes_original.append(box)

        # assert
        box_data = object_info["box"]
        assert box_data == box


    x_i_int_unscaled_ret = np.array(x_i_int_unscaled_ret)
    x_i_int_ret = np.array(x_i_int_ret)
    x_i_int_out_unscaled_ret = np.array(x_i_int_out_unscaled_ret)
    x_i_int_out_ret = np.array(x_i_int_out_ret)
    return categories, \
           ds_categories, \
           boxes_original, \
           boxes_unscaled, \
           x_i_int_unscaled_ret, \
           x_i_int_ret, \
           x_i_int_out_unscaled_ret, \
           x_i_int_out_ret


def visualize_image(img_to_show,
                    x_i_proper,
                    x_i_out,
                    rectangles,
                    title=None,
                    save_path=None):
    rectangles = [r for r in rectangles if len(r) > 0]
    rectangles = np.array(rectangles)
    _, ax = plt.subplots(1, 1)
    # ax.set_axis_off()
    ax.imshow(img_to_show)
    if len(x_i_proper) > 0:
        ax.plot(x_i_proper[:, 0], x_i_proper[:, 1], "bx", markersize="10", markeredgewidth=2)
    if len(x_i_out) > 0:
        ax.plot(x_i_out[:, 0], x_i_out[:, 1], "rx", markersize="10", markeredgewidth=2)
    for rect in rectangles:
        ax.plot(np.hstack((rect[:, 1], rect[0, 1])), np.hstack((rect[:, 0], rect[0, 0])), "y-.", linewidth=2)

    plt.title(title)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    show = True
    if show:
        plt.show()
    plt.close()


def compute():

    task = "panoptic"
    args = scene_args()

    start_time = time.time()
    data_entries, ready_entries, min_counts_map, config_read = get_cached_data(args.conf_base_path,
                                                                               format_suffix=ConfStatic.toml_suffix,
                                                                               out_log=True)

    for e_i, config_entry in enumerate(data_entries):

        if config_entry.__contains__(BOXES_2D_KEY):
            continue
        else:
            ready_entries += 1
            if args.max_entries and ready_entries > args.max_entries:
                break

        # START FUNCTION
        # segments_info, segmentation, scale = fc(img_path, simple_path_prefix)

        # a) read imgs, segmentation, segm_info
        # i) infer the data (possibly read the data and assert ==)
        # ii) read the data
        segments_info, segmentation_map, segm_contrast_img, segm_vis_img, original_img = \
            get_imgs(config_entry, args.out_data_root)

        # b) scale (merge?)
        scale = get_scale(original_img, segmentation_map)

        # c) all boxes/x_is...
        categories, \
        ds_categories, \
        boxes_original, \
        boxes_unscaled, \
        x_i_int_unscaled, \
        x_i_int, \
        x_i_int_out_unscaled, \
        x_i_int_out = get_boxes(config_entry, segments_info, segmentation_map, scale)

        # d) visualization
        path_pref = get_simple_path_prefix(config_entry, args.out_data_root)
        title = ", ".join([f"{dsc}->{c}" for dsc, c in zip(ds_categories, categories)])
        title += f"\n ds categories: {ds_categories}"
        visualize_image(segm_contrast_img,
                        x_i_int_unscaled,
                        x_i_int_out_unscaled,
                        boxes_unscaled,
                        title=title,
                        save_path=f"{path_pref}_segmentation_contrast_boxes.png")
        visualize_image(segm_vis_img,
                        x_i_int_unscaled,
                        x_i_int_out_unscaled,
                        boxes_unscaled,
                        title=title)
        # visualize_image(original_img,
        #                 x_i_int,
        #                 x_i_int_out,
        #                 boxes_original,
        #                 title=title)
        # END FUNCTION

        # e) write to orig config (anything else?)
        config_entry[BOXES_2D_KEY] = boxes_original

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
    # assert
    for k, v in stuff_dataset_id_to_contiguous_id.items():
        assert k == v
    for k, v in thing_dataset_id_to_contiguous_id.items():
        assert k == v
    compute()

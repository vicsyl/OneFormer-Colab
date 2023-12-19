import time
from collections import defaultdict

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from pyquaternion import Quaternion

from common.common_transforms import project_metropolis
from common.data_parsing import save, get_cached_data, Configuration, read_toml
from common.fitting import fit_min_area_rect
from common.vanishing_point import change_x_3d_arkit, change_r_arkit
from metadata import *
from set_args import scene_args

BOXES_2D_KEY_PROJECTION = "segmented_boxes_2d_projection"
BOXES_2D_KEY_CLASSES = "segmented_boxes_2d_classes"


def simple_show_img(img_l, title=None):
    plt.figure()
    plt.title(title)
    plt.imshow(img_l)
    plt.show()
    plt.close()


def get_scale(original_img, segmentation_map):
    or_size = original_img.shape[:2]
    size = segmentation_map.shape[:2]
    scale = float(or_size[0]) / float(size[0])
    scale2 = float(or_size[1]) / float(size[1])
    # MAY not hold actually
    assert np.isclose(scale, scale2)
    return scale


def get_simple_path_prefix(config_entry, out_data_root):

    img_path = config_entry['orig_file_path']
    # hack - ../../download/... -> ./download/.
    img_path = img_path[4:]
    # print(f"path: {img_path}")
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


def get_object_info_category(object_info):
    # isthing = object_info["isthing"]
    # well it is just using stuff IMHO
    id_map, classes_map = if_thing_stuff_maps[False]
    id = id_map[object_info["category_id"]]
    assert id == object_info["category_id"]
    category = classes_map[id]
    return category


def get_boxes_based_on_projection(config_entry, segments_info, segmentation_map, scale):

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

        if coord_x_y[0] < 0 or coord_x_y[0] >= segmentation_map.shape[1] or \
                coord_x_y[1] < 0 or coord_x_y[1] >= segmentation_map.shape[0]:
            clean()
            continue
        obj_idx = segmentation_map[coord_x_y[1], coord_x_y[0]]
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
        object_info = segments_info["objects"][obj_idx - 1]
        assert object_info["id"] == obj_idx

        category = get_object_info_category(object_info)
        categories.append(category)

        pixels_to_fit = np.array(np.where(segmentation_map == obj_idx)).T
        pixels_to_fit = pixels_to_fit[:, [1, 0]]
        box = fit_min_area_rect(pixels_to_fit)
        boxes_unscaled.append(box.tolist().copy())
        box *= scale
        box = box.tolist()
        boxes_original.append(box)

    x_i_int_unscaled_ret = np.array(x_i_int_unscaled_ret)
    x_i_int_ret = np.array(x_i_int_ret)
    x_i_int_out_unscaled_ret = np.array(x_i_int_out_unscaled_ret)
    x_i_int_out_ret = np.array(x_i_int_out_ret)
    return categories, \
           config_entry['names'], \
           boxes_original, \
           boxes_unscaled, \
           x_i_int_unscaled_ret, \
           x_i_int_ret, \
           x_i_int_out_unscaled_ret, \
           x_i_int_out_ret


def visualize_image(args,
                    img_to_show,
                    x_i_proper,
                    x_i_out,
                    rectangles,
                    title=None,
                    save_path=None):
    rectangles = [r for r in rectangles if len(r) > 0]
    rectangles = np.array(rectangles)
    _, ax = plt.subplots(1, 1)
    ax.set_xlim(0, img_to_show.shape[1])
    ax.set_ylim(img_to_show.shape[0], 0)
    # ax.set_axis_off()
    ax.imshow(img_to_show)
    if len(x_i_proper) > 0:
        ax.plot(x_i_proper[:, 0], x_i_proper[:, 1], "bx", markersize="10", markeredgewidth=2)
    if len(x_i_out) > 0:
        ax.plot(x_i_out[:, 0], x_i_out[:, 1], "rx", markersize="10", markeredgewidth=2)
    for rect in rectangles:
        ax.plot(np.hstack((rect[:, 0], rect[0, 0])), np.hstack((rect[:, 1], rect[0, 1])), "r-.", linewidth=2)

    plt.title(title, fontsize="x-small")
    if args.save and save_path:
        plt.savefig(save_path) #, bbox_inches="tight")
    if args.show:
        plt.show()
    plt.close()


def get_or_infer(args, config_entry):

    if args.infer:
        from infer import infer_and_save
        segments_info, segmentation_map, segm_contrast_img, segm_vis_img, original_img = \
            infer_and_save(config_entry, args.out_data_root)
        if args.infer_test:
            segments_info2, segmentation_map2, segm_contrast_img2, segm_vis_img2, original_img2 = \
                get_imgs(config_entry, args.out_data_root)
            assert segments_info == segments_info2
            assert np.all(segmentation_map == segmentation_map2)
            assert np.all(segm_contrast_img == segm_contrast_img2)
            assert np.all(segm_vis_img == segm_vis_img2)
            assert np.all(original_img == original_img2)
    else:
        segments_info, segmentation_map, segm_contrast_img, segm_vis_img, original_img = \
            get_imgs(config_entry, args.out_data_root)

    return segments_info, segmentation_map, segm_contrast_img, segm_vis_img, original_img


def get_boxes_based_on_classes(config_entry,
                               segments_info,
                               segmentation_map,
                               original_img):

    # object_info = "{'id': 1,
    # 'isthing': False,
    # 'category_id': 0,
    # 'area': 112475.0,
    # 'box': list[Float, 4, 2]}"

    # segm. category -> list of object infos
    segmentation_category_map = defaultdict(list)
    for object_info in segments_info["objects"]:
        category = get_object_info_category(object_info)
        segmentation_category_map[category].append(object_info["id"])

    # unique object info or None of not unique
    object_infos = []
    first_ds_category_id = {}
    for i, name in enumerate(config_entry["names"]):
        if first_ds_category_id.__contains__(name):
            object_infos[first_ds_category_id[name]] = None
            object_infos.append(None)
            continue
        first_ds_category_id[name] = i

        # arkit_class_names_map[name]: seg.category => segmented ids
        looking_for = arkit_class_names_map[name]
        all_idss_found = []
        for lfc in looking_for:
            all_idss_found.extend(segmentation_category_map[lfc])
            if len(all_idss_found) > 1:
                break
        if len(all_idss_found) == 1:
            sid = all_idss_found[0]
            object_info = segments_info["objects"][sid - 1]
            assert object_info["id"] == sid
            object_infos.append(object_info)
        else:
            object_infos.append(None)

    # scale
    scale = get_scale(original_img, segmentation_map)
    boxes_unscaled = []
    boxes_original = []
    categories = []
    # ds_categories = config_entry['names']

    for object_info in object_infos:
        if object_info is None:
            boxes_unscaled.append([])
            boxes_original.append([])
            categories.append("not found")
            continue
        sid = object_info["id"]
        pixels_to_fit = np.array(np.where(segmentation_map == sid)).T
        pixels_to_fit = pixels_to_fit[:, [1, 0]]
        box = fit_min_area_rect(pixels_to_fit)
        boxes_unscaled.append(box.tolist().copy())
        box *= scale
        box = box.tolist()
        boxes_original.append(box)

        category = get_object_info_category(object_info)
        categories.append(category)

    return categories, \
           config_entry['names'], \
           boxes_original, \
           boxes_unscaled


def split_str(s, max_row_length):

    ret = ""
    for i in range(max(1, (len(s) - 1) // max_row_length + 1)):
        beg = i * max_row_length
        end = (i + 1) * max_row_length
        ret += f"\n{s[beg:end]}"
    return ret


def compute_boxes_based_on_classes(config_entry,
                                   segments_info,
                                   segmentation_map,
                                   segm_contrast_img,
                                   segm_vis_img,
                                   original_img,
                                   args):

    categories, ds_categories, boxes_original, boxes_unscaled = \
        get_boxes_based_on_classes(config_entry,
                                   segments_info,
                                   segmentation_map,
                                   original_img)

    path_pref = get_simple_path_prefix(config_entry, args.out_data_root)
    mappings = ", ".join([f"{dsc}->{c}" for dsc, c in zip(ds_categories, categories)])
    # print(f"mappings: {mappings}")
    title = f"based on classes: map[arkit -> segmentation]: {mappings}"
    title += f"\n ds categories: {ds_categories}"

    segmentation_categories = [get_object_info_category(oi) for oi in segments_info["objects"]]

    s = f"segm. categories: {segmentation_categories}"
    title += split_str(s, max_row_length=130)
    # title += str(boxes_unscaled)
    visualize_image(args,
                    segm_contrast_img,
                    [],
                    [],
                    boxes_unscaled,
                    title=title,
                    save_path=f"{path_pref}_segmentation_contrast_boxes_classes.png")
    # visualize_image(args,
    #                 segm_vis_img,
    #                 [],
    #                 [],
    #                 boxes_unscaled,
    #                 title=title)
    # visualize_image(args,
    #                 original_img,
    #                 [],
    #                 [],
    #                 boxes_original,
    #                 title=title)

    # write to orig config
    config_entry[BOXES_2D_KEY_CLASSES] = boxes_original


def compute_boxes_based_on_projection(config_entry,
                                      segments_info,
                                      segmentation_map,
                                      segm_contrast_img,
                                      segm_vis_img,
                                      original_img,
                                      args):

    # scale
    scale = get_scale(original_img, segmentation_map)

    # all boxes/x_is...
    categories, \
    ds_categories, \
    boxes_original, \
    boxes_unscaled, \
    x_i_int_unscaled, \
    x_i_int, \
    x_i_int_out_unscaled, \
    x_i_int_out = get_boxes_based_on_projection(config_entry, segments_info, segmentation_map, scale)

    # visualization
    path_pref = get_simple_path_prefix(config_entry, args.out_data_root)
    title = "based on projection: map[arkit -> segmentation]:" + ", ".join([f"{dsc}->{c}" for dsc, c in zip(ds_categories, categories)])
    title += f"\n ds categories: {ds_categories}"
    segmentation_categories = [get_object_info_category(oi) for oi in segments_info["objects"]]
    s = f"segm. categories: {segmentation_categories}"
    title += split_str(s, max_row_length=130)
    # title += str(boxes_unscaled)
    visualize_image(args,
                    segm_contrast_img,
                    x_i_int_unscaled,
                    x_i_int_out_unscaled,
                    boxes_unscaled,
                    title=title,
                    save_path=f"{path_pref}_segmentation_contrast_boxes_x_gt.png")
    # visualize_image(args,
    #                 segm_vis_img,
    #                 x_i_int_unscaled,
    #                 x_i_int_out_unscaled,
    #                 boxes_unscaled,
    #                 title=title)
    # visualize_image(args,
    #                 original_img,
    #                 x_i_int,
    #                 x_i_int_out,
    #                 boxes_original,
    #                 title=title)

    # write to orig config
    config_entry[BOXES_2D_KEY_PROJECTION] = boxes_original


def compute():

    args = scene_args()

    start_time = time.time()
    data_entries, or_ready_entries, min_counts_map, config_read = get_cached_data(args.conf_base_path,
                                                                               format_suffix=Configuration.toml_suffix,
                                                                               out_log=True)

    ready_entries = or_ready_entries
    for e_i, config_entry in enumerate(data_entries):

        args.save = (3000 < e_i < 5000)

        if config_entry.__contains__(BOXES_2D_KEY_PROJECTION) and config_entry.__contains__(BOXES_2D_KEY_CLASSES):
        # if config_entry.__contains__(BOXES_2D_KEY_CLASSES):
            continue
        else:
            if args.max_entries and ready_entries > args.max_entries:
                break

        # get or infer
        segments_info, \
        segmentation_map, \
        segm_contrast_img, \
        segm_vis_img, \
        original_img = get_or_infer(args, config_entry)

        compute_boxes_based_on_projection(config_entry,
                                          segments_info,
                                          segmentation_map,
                                          segm_contrast_img,
                                          segm_vis_img,
                                          original_img,
                                          args)

        compute_boxes_based_on_classes(config_entry,
                                       segments_info,
                                       segmentation_map,
                                       segm_contrast_img,
                                       segm_vis_img,
                                       original_img,
                                       args)

        if ready_entries % 1000 == 0:
            elapased = time.time() - start_time
            print(f"ready_entries: {ready_entries}")
            print(f"time elapsed: %f sec" % elapased)

        if ready_entries % args.cache_every_other == 0 and ready_entries != or_ready_entries:
            sp_file_path = f"{args.conf_base_path}_sp={ready_entries}"
            # FIXME: there is still a problem
            #   for cache_every_other == 1000 it will first save with 1001 entries ready
            save(f"{sp_file_path}{args.format_suffix}",
                 data_entries,
                 objects_counts_map={},
                 at_least_objects_counts_map=min_counts_map,
                 conf_attribute_map=config_read)
        if e_i != len(data_entries) - 1:
            ready_entries += 1

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

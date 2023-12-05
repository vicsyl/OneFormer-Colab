# Setup detectron2 logger
import argparse
import os
import sys
import time

from common.data_parsing import get_cached_data, ConfStatic, save
from data_save import save_data
from set_args import scene_args

sys.path.insert(0, os.path.abspath('../detectron2'))
#sys.path.insert(0, os.path.abspath('./demo'))

from detectron2.utils.logger import setup_logger
setup_logger()
setup_logger(name="oneformer")

# Import libraries
import numpy as np
import cv2
import torch
# from google.colab.patches import cv2_imshow
import imutils

# Import detectron2 utilities
from detectron2.config import get_cfg
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.data import MetadataCatalog
from demo.defaults import DefaultPredictor
from demo.visualizer import Visualizer, ColorMode

# import OneFormer Project
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)

# cpu_device = torch.device("cpu")
cpu_device = torch.device("cuda")


SWIN_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_swin_large_IN21k_384_bs16_90k.yaml",
            "coco": "configs/coco/oneformer_swin_large_IN21k_384_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_swin_large_IN21k_384_bs16_160k.yaml",}


DINAT_CFG_DICT = {"cityscapes": "configs/cityscapes/oneformer_dinat_large_bs16_90k.yaml",
            "coco": "configs/coco/oneformer_dinat_large_bs16_100ep.yaml",
            "ade20k": "configs/ade20k/oneformer_dinat_large_IN21k_384_bs16_160k.yaml",}


def setup_cfg(dataset, model_path, use_swin):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    if use_swin:
      cfg_path = SWIN_CFG_DICT[dataset]
    else:
      cfg_path = DINAT_CFG_DICT[dataset]
    cfg.merge_from_file(cfg_path)
    cfg.MODEL.DEVICE = 'cuda'
    cfg.MODEL.WEIGHTS = model_path
    cfg.freeze()
    return cfg


def setup_modules(dataset, model_path, use_swin):
    cfg = setup_cfg(dataset, model_path, use_swin)
    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(
        cfg.DATASETS.TEST_PANOPTIC[0] if len(cfg.DATASETS.TEST_PANOPTIC) else "__unused"
    )
    if 'cityscapes_fine_sem_seg_val' in cfg.DATASETS.TEST_PANOPTIC[0]:
        from cityscapesscripts.helpers.labels import labels
        stuff_colors = [k.color for k in labels if k.trainId != 255]
        metadata = metadata.set(stuff_colors=stuff_colors)

    return predictor, metadata


def panoptic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "panoptic")
    panoptic_seg, segments_info = predictions["panoptic_seg"]
    out = visualizer.draw_panoptic_seg_predictions(
        panoptic_seg.to(cpu_device), segments_info, alpha=0.5
    )
    return predictions, out


def instance_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "instance")
    instances = predictions["instances"].to(cpu_device)
    out = visualizer.draw_instance_predictions(predictions=instances, alpha=0.5)
    return predictions, out


def semantic_run(img, predictor, metadata):
    visualizer = Visualizer(img[:, :, ::-1], metadata=metadata, instance_mode=ColorMode.IMAGE)
    predictions = predictor(img, "semantic")
    out = visualizer.draw_sem_seg(
        predictions["sem_seg"].argmax(dim=0).to(cpu_device), alpha=0.5
    )
    return predictions, out

TASK_INFER = {"panoptic": panoptic_run,
              "instance": instance_run,
              "semantic": semantic_run}


use_swin = True

# download model checkpoint
import os
import subprocess
if not use_swin:
  if not os.path.exists("250_16_dinat_l_oneformer_ade20k_160k.pth"):
    subprocess.run('wget https://shi-labs.com/projects/oneformer/ade20k/250_16_dinat_l_oneformer_ade20k_160k.pth', shell=True)
  predictor, metadata = setup_modules("ade20k", "250_16_dinat_l_oneformer_ade20k_160k.pth", use_swin)
else:
  if not os.path.exists("250_16_swin_l_oneformer_ade20k_160k.pth"):
    subprocess.run('wget https://shi-labs.com/projects/oneformer/ade20k/250_16_swin_l_oneformer_ade20k_160k.pth', shell=True)
  predictor, metadata = setup_modules("ade20k", "250_16_swin_l_oneformer_ade20k_160k.pth", use_swin)
print(f"METADATA: {metadata}")


BOXES_2D_KEY = "segmented_boxes_2d"
WIDTHS_HEIGHTS_KEY = "segmented_wh"


def infer(config_entry, out_data_root):

    task = "panoptic"
    img_path = config_entry['orig_file_path']

    # hack - ../../download/... -> ./download/.
    img_path = img_path[4:]

    print(f"path: {img_path}")
    original_img = cv2.imread(img_path)
    or_size = original_img.shape[:2]

    resized_img = imutils.resize(original_img, width=640)
    size = resized_img.shape[:2]

    scale_to_or_0 = float(or_size[0]) / float(size[0])
    scale_to_or_1 = float(or_size[1]) / float(size[1])
    print(f"scale_to_or_0: {scale_to_or_0}")
    print(f"scale_to_or_1: {scale_to_or_1}")

    # MAY not hold actually
    assert np.isclose(scale_to_or_0, scale_to_or_1)

    predictions, out = TASK_INFER[task](resized_img, predictor, metadata)
    segm_vis_img = out.get_image()
    segmentation_map, segments_info = predictions["panoptic_seg"]

    assumed_prefix_l = len("./download/3dod/Training/")
    simple_path_prefix = f"{out_data_root}/{img_path[assumed_prefix_l:]}"[:-4]

    segmentation_map = segmentation_map.cpu().numpy()
    segments_info = save_data(simple_path_prefix, segmentation_map, segments_info, scale_to_or_0, segm_vis_img)

    contrast = 230 // segmentation_map.max()
    segm_contrast_img = segmentation_map * contrast
    segm_contrast_img[segm_contrast_img == 0] = 255

    return segments_info, segmentation_map, segm_contrast_img, segm_vis_img, original_img


def loop_compute():

    args = scene_args()

    # toml_conf = read_toml(args.conf_base_path)
    # configs/ARKitScenes=obj=2_max=100_sp=10.toml => conf_base_path = configs/ARKitScenes=obj=2_max=100

    start_time = time.time()
    data_entries, ready_entries, min_counts_map, config_read = get_cached_data(args.conf_base_path,
                                                                               format_suffix=ConfStatic.toml_suffix,
                                                                               out_log=True)

    for e_i, config_entry in enumerate(data_entries):

        if config_entry.__contains__(BOXES_2D_KEY):
            continue
        else:
            ready_entries += 1
            # if ready_entries > 10:
            #     break

        segments_info, segmentation_map, segm_contrast_img, segm_vis_img, original_img = \
            infer(config_entry, args.out_data_root)

        if ready_entries % args.cache_every_other == 0:
            sp_file_path = f"{args.conf_base_path}_sp={ready_entries}{args.format_suffix}"
            save(sp_file_path,
                 data_entries,
                 objects_counts_map={},
                 at_least_objects_counts_map=min_counts_map,
                 conf_attribute_map=config_read)

    print("Saving the final file")
    elapased = time.time() - start_time
    print(f"Total time: %f sec" % elapased)

    sp_file_path = f"{args.conf_base_path}_sp={ready_entries}{args.format_suffix}"
    save(sp_file_path,
         data_entries,
         objects_counts_map={},
         at_least_objects_counts_map=min_counts_map,
         conf_attribute_map=config_read)


if __name__ == "__main__":
    loop_compute()

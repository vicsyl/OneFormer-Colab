import os
import os.path as osp
from pathlib import Path


def infer_me(in_dir, out_dir):

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(in_dir):
        full_path = f"{in_dir}/{file_name}"
        if os.path.isfile(full_path) and file_name.endswith(".jpg"):
            print(f"{full_path} => {out_dir}/{file_name}")
            infer = True
            if not infer:
                continue

            from local_infer_nn import infer_and_save
            infer_and_save(in_dir, file_name, out_dir)


def only_dirs(d):
    return [_ for _ in os.listdir(d) if osp.isdir(osp.join(d, _))]


def compute(root_in_dir, root_out_dir="./twelve_scenes_infer_layout"):

    # LAYOUT:
    # root/scene(apt1)/room(kitchen)/info.txt

    scenes = only_dirs(root_in_dir)
    # scenes = ["apt1"]
    scene_map = {}
    for scene in scenes:
        scene_map[scene] = []
        rooms = only_dirs(osp.join(root_in_dir, scene))
        # rooms = ["kitchen"]
        for room in rooms:
            in_path = os.path.join(root_in_dir, scene, room)
            out_dir = os.path.join(root_out_dir, scene, room)
            print(f"{in_path} => {out_dir}")
            infer_me(in_dir=in_path, out_dir=out_dir)


if __name__ == "__main__":
    # compute("../twelve_scenes/data/ds")
    compute("../twelve_scenes")

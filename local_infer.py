import os
from pathlib import Path


def infer_me(in_dir, out_dir):

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    for file_name in os.listdir(in_dir):
        full_path = f"{in_dir}/{file_name}"
        if os.path.isfile(full_path) and file_name.endswith(".jpg"):
            print(f"{full_path} => {out_dir}/{file_name}")
            infer = False
            if not infer:
                continue

            from local_infer_nn import infer_and_save
            infer_and_save(in_dir, file_name, out_dir)


if __name__ == "__main__":
    infer_me(in_dir="../twelve_scenes/kitchen/data/", out_dir="./twelve_scenes_infer/1")


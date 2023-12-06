import argparse

from common.data_parsing import ConfStatic


def scene_args():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_data_root",
        default="./out_data",
        help="output folder"
    )
    parser.add_argument('--infer', action='store_true', default=True)
    parser.add_argument('--no-infer', dest='infer', action='store_false')
    parser.add_argument('--infer_test', action='store_true', default=False)
    parser.add_argument('--no-infer_test', dest='infer_test', action='store_false')
    parser.add_argument("--max_entries", type=int, default=None)
    # configs/ARKitScenes=obj=2_max=100_sp=10.toml => conf_base_path = configs/ARKitScenes=obj=2_max=100
    parser.add_argument("--conf_base_path", type=str, default=None)
    parser.add_argument("--cache_every_other", type=int, default=5000)
    parser.add_argument("--format_suffix", type=str, default=ConfStatic.toml_suffix)
    args = parser.parse_args()
    print(f"Args: {args}")
    return args



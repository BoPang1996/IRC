import os
import argparse
import os
import numpy as np
from PIL import Image
from tqdm import tqdm
from multiprocessing import Pool
from skimage.segmentation import slic
from nuscenes.nuscenes import NuScenes
from functools import partial


def compute_slic(cam_token, set_path):
    cam = nusc.get("sample_data", cam_token)
    im = Image.open(os.path.join(nusc.dataroot, cam["filename"]))
    segments_slic = slic(
        im, n_segments=150, compactness=6, sigma=3.0, start_label=0
    ).astype(np.uint8)
    im = Image.fromarray(segments_slic)
    im.save(os.path.join(set_path,
        "superpixels/superpixels_slic/" + cam["token"] + ".png")
    )


def compute_slic_30(cam_token, set_path):
    cam = nusc.get("sample_data", cam_token)
    im = Image.open(os.path.join(nusc.dataroot, cam["filename"]))
    segments_slic = slic(
        im, n_segments=30, compactness=6, sigma=3.0, start_label=0
    ).astype(np.uint8)
    im = Image.fromarray(segments_slic)
    im.save(os.path.join(set_path,
        "superpixels/superpixels_slic_30/" + cam["token"] + ".png")
    )


if __name__ == "__main__":
    nuscenes_path = "/home/zy/nuScenes"
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--model", type=str, default="minkunet", help="specify the model targeted, either minkunet or voxelnet"
    )
    assert os.path.exists(nuscenes_path), f"nuScenes not found in {nuscenes_path}"
    args = parser.parse_args()
    assert args.model in ["minkunet", "voxelnet"]
    nusc = NuScenes(
        version="v1.0-trainval", dataroot=nuscenes_path, verbose=False
    )
    os.makedirs(os.path.join(nuscenes_path, "superpixels/superpixels_slic/"), exist_ok=True)
    os.makedirs(os.path.join(nuscenes_path, "superpixels/superpixels_slic_30/"), exist_ok=True)
    camera_list = [
        "CAM_FRONT",
        "CAM_FRONT_RIGHT",
        "CAM_BACK_RIGHT",
        "CAM_BACK",
        "CAM_BACK_LEFT",
        "CAM_FRONT_LEFT",
    ]
    with Pool(6) as p:
        for scene_idx in tqdm(range(len(nusc.scene))):
            scene = nusc.scene[scene_idx]
            current_sample_token = scene["first_sample_token"]
            while current_sample_token != "":
                current_sample = nusc.get("sample", current_sample_token)
                if args.model == "minkunet":
                    func = partial(compute_slic, set_path=nuscenes_path)
                elif args.model == "voxelnet":
                    func = partial(compute_slic_30, set_path=nuscenes_path)
                p.map(
                    func,
                    [
                        current_sample["data"][camera_name]
                        for camera_name in camera_list
                    ],
                )
                current_sample_token = current_sample["next"]

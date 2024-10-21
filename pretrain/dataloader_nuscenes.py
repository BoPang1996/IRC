import os
import copy
import torch
import numpy as np
from PIL import Image
import MinkowskiEngine as ME
from pyquaternion import Quaternion
from torch.utils.data import Dataset
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.utils.splits import create_splits_scenes
from nuscenes.utils.data_classes import LidarPointCloud


CUSTOM_SPLIT = [
    "scene-0008", "scene-0009", "scene-0019", "scene-0029", "scene-0032", "scene-0042",
    "scene-0045", "scene-0049", "scene-0052", "scene-0054", "scene-0056", "scene-0066",
    "scene-0067", "scene-0073", "scene-0131", "scene-0152", "scene-0166", "scene-0168",
    "scene-0183", "scene-0190", "scene-0194", "scene-0208", "scene-0210", "scene-0211",
    "scene-0241", "scene-0243", "scene-0248", "scene-0259", "scene-0260", "scene-0261",
    "scene-0287", "scene-0292", "scene-0297", "scene-0305", "scene-0306", "scene-0350",
    "scene-0352", "scene-0358", "scene-0361", "scene-0365", "scene-0368", "scene-0377",
    "scene-0388", "scene-0391", "scene-0395", "scene-0413", "scene-0427", "scene-0428",
    "scene-0438", "scene-0444", "scene-0452", "scene-0453", "scene-0459", "scene-0463",
    "scene-0464", "scene-0475", "scene-0513", "scene-0533", "scene-0544", "scene-0575",
    "scene-0587", "scene-0589", "scene-0642", "scene-0652", "scene-0658", "scene-0669",
    "scene-0678", "scene-0687", "scene-0701", "scene-0703", "scene-0706", "scene-0710",
    "scene-0715", "scene-0726", "scene-0735", "scene-0740", "scene-0758", "scene-0786",
    "scene-0790", "scene-0804", "scene-0806", "scene-0847", "scene-0856", "scene-0868",
    "scene-0882", "scene-0897", "scene-0899", "scene-0976", "scene-0996", "scene-1012",
    "scene-1015", "scene-1016", "scene-1018", "scene-1020", "scene-1024", "scene-1044",
    "scene-1058", "scene-1094", "scene-1098", "scene-1107",
]


def minkunet_collate_pair_fn(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    (
        coords,
        feats,
        coords2,
        feats2,
        ts_coords,
        ts_feats,
        images,
        pairing_points,
        # pairing_points_ts,
        pairing_images,
        # pairing_images_ts,
        inverse_indexes,
        superpixels,
        # superpixels_ts,
    ) = list(zip(*list_data))
    batch_n_points, batch_n_pairings = [], []
    ts_batch_n_points = []
    # batch_n_pairings_ts = []

    offset = 0
    pairing_points_org = copy.deepcopy(pairing_points)
    pairing_images_org = copy.deepcopy(pairing_images)

    # offset_ts = 0
    # pairing_points_ts_org = copy.deepcopy(pairing_points_ts)
    # pairing_images_ts_org = copy.deepcopy(pairing_images_ts)

    for batch_id in range(len(coords)):

        # Move batchids to the beginning
        coords[batch_id][:, 0] = batch_id
        coords2[batch_id][:, 0] = batch_id
        ts_coords[batch_id][:, 0] = batch_id
        pairing_points[batch_id][:] += offset
        # pairing_points_ts[batch_id][:] += offset_ts
        pairing_images[batch_id][:, 0] += batch_id * images[0].shape[0]
        # pairing_images_ts[batch_id][:, 0] += batch_id * images[0].shape[0]

        batch_n_points.append(coords[batch_id].shape[0])
        ts_batch_n_points.append(ts_coords[batch_id].shape[0])
        batch_n_pairings.append(pairing_points[batch_id].shape[0])
        # batch_n_pairings_ts.append(pairing_points_ts[batch_id].shape[0])
        offset += coords[batch_id].shape[0]
        # offset_ts += ts_coords[batch_id].shape[0]

    # Concatenate all lists
    coords_batch = torch.cat(coords, 0).int()
    coords2_batch = torch.cat(coords2, 0).int()
    ts_coords_batch = torch.cat(ts_coords, 0).int()
    pairing_points = torch.tensor(np.concatenate(pairing_points))
    # pairing_points_ts = torch.tensor(np.concatenate(pairing_points_ts))
    pairing_images = torch.tensor(np.concatenate(pairing_images))
    # pairing_images_ts = torch.tensor(np.concatenate(pairing_images_ts))
    feats_batch = torch.cat(feats, 0).float()
    feats2_batch = torch.cat(feats2, 0).float()
    ts_feats_batch = torch.cat(ts_feats, 0).float()
    images_batch = torch.cat(images, 0).float()
    superpixels_batch = torch.tensor(np.concatenate(superpixels))
    # superpixels_ts_batch = torch.tensor(np.concatenate(superpixels_ts))
    return {
        "sinput_C": coords_batch,
        "sinput_C2": coords2_batch,
        "sinput_F": feats_batch,
        "sinput_F2": feats2_batch,
        "sinput_ts_C": ts_coords_batch,
        "sinput_ts_F": ts_feats_batch,
        "input_I": images_batch,
        "pairing_points_org": pairing_points_org,
        # "pairing_points_ts_org": pairing_points_ts_org,
        "pairing_images_org": pairing_images_org,
        # "pairing_images_ts_org": pairing_images_ts_org,
        "pairing_points": pairing_points,
        # "pairing_points_ts": pairing_points_ts,
        "pairing_images": pairing_images,
        # "pairing_images_ts": pairing_images_ts,
        "batch_n_points": batch_n_points,
        "ts_batch_n_points": ts_batch_n_points,
        "batch_n_pairings": batch_n_pairings,
        # "batch_n_pairings_ts": batch_n_pairings_ts,
        "inverse_indexes": inverse_indexes,
        "superpixels": superpixels_batch,
        # "superpixels_ts": superpixels_ts_batch,
    }


class NuScenesMatchDataset(Dataset):
    """
    Dataset matching a 3D points cloud and an image using projection.
    """

    def __init__(
        self,
        phase,
        config,
        shuffle=False,
        cloud_transforms=None,
        mixed_transforms=None,
        **kwargs,
    ):
        self.config = config
        self.phase = phase
        self.shuffle = shuffle
        self.cloud_transforms = cloud_transforms
        self.mixed_transforms = mixed_transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]
        self.superpixels_type = config["superpixels_type"]
        self.bilinear_decoder = config["decoder"] == "bilinear"

        if "cached_nuscenes" in kwargs:
            self.nusc = kwargs["cached_nuscenes"]
        else:
            self.nusc = NuScenes(
                version="v1.0-trainval", dataroot=config["data_root"], verbose=False
            )

        self.list_keyframes = []
        # a skip ratio can be used to reduce the dataset size and accelerate experiments
        try:
            skip_ratio = config["dataset_skip_step"]
        except KeyError:
            skip_ratio = 1
        skip_counter = 0
        if phase in ("train", "val", "test"):
            phase_scenes = create_splits_scenes()[phase]
        elif phase == "parametrizing":
            phase_scenes = list(
                set(create_splits_scenes()["train"]) - set(CUSTOM_SPLIT)
            )
        elif phase == "verifying":
            phase_scenes = CUSTOM_SPLIT
        # create a list of camera & lidar scans
        for scene_idx in range(len(self.nusc.scene)):
            scene = self.nusc.scene[scene_idx]
            if scene["name"] in phase_scenes:
                skip_counter += 1
                if skip_counter % skip_ratio == 0:
                    self.create_list_of_scans(scene)

    def create_list_of_scans(self, scene):
        # Get first and last keyframe in the scene
        current_sample_token = scene["first_sample_token"]

        # Loop to get all successive keyframes
        list_data = []
        while current_sample_token != "":
            current_sample = self.nusc.get("sample", current_sample_token)
            list_data.append(current_sample["data"])
            current_sample_token = current_sample["next"]

        # Add new scans in the list
        self.list_keyframes.extend(list_data)

    def map_pointcloud_to_image(self, data, min_dist: float = 1.0):
        """
        Given a lidar token and camera sample_data token, load pointcloud and map it to
        the image plane. Code adapted from nuscenes-devkit
        https://github.com/nutonomy/nuscenes-devkit.
        :param min_dist: Distance from the camera below which points are discarded.
        """
        pointsensor = self.nusc.get("sample_data", data["LIDAR_TOP"])
        pcl_path = os.path.join(self.nusc.dataroot, pointsensor["filename"])
        pc_original = LidarPointCloud.from_file(pcl_path)
        pc_ref = pc_original.points

        # get the temporal shift pc
        current_sample = self.nusc.get("sample", pointsensor["sample_token"])
        ts_sample_token = current_sample['next'] if current_sample['next'] != '' else current_sample['prev']
        ts_sample = self.nusc.get("sample", ts_sample_token)
        ts_data = ts_sample['data']
        ts_pointsensor = self.nusc.get("sample_data", ts_data["LIDAR_TOP"])
        ts_pcl_path = os.path.join(self.nusc.dataroot, ts_pointsensor["filename"])
        ts_pc_original = LidarPointCloud.from_file(ts_pcl_path)
        ts_pc_ref = ts_pc_original.points

        images = []
        superpixels = []
        pairing_points = np.empty(0, dtype=np.int64)
        pairing_images = np.empty((0, 3), dtype=np.int64)
        camera_list = [
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ]
        if self.shuffle:
            np.random.shuffle(camera_list)
        for i, camera_name in enumerate(camera_list):
            pc = copy.deepcopy(pc_original)
            cam = self.nusc.get("sample_data", data[camera_name])
            im = np.array(Image.open(os.path.join(self.nusc.dataroot, cam["filename"])))
            sp = Image.open(os.path.join(self.config["data_root"],
                f"superpixels/"
                f"superpixels_{self.superpixels_type}/{cam['token']}.png")
            )
            superpixels.append(np.array(sp))

            # Points live in the point sensor frame. So they need to be transformed via
            # global to the image plane.
            # First step: transform the pointcloud to the ego vehicle frame for the
            # timestamp of the sweep.
            cs_record = self.nusc.get(
                "calibrated_sensor", pointsensor["calibrated_sensor_token"]
            )
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix)
            pc.translate(np.array(cs_record["translation"]))

            # Second step: transform from ego to the global frame.
            poserecord = self.nusc.get("ego_pose", pointsensor["ego_pose_token"])
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix)
            pc.translate(np.array(poserecord["translation"]))

            # Third step: transform from global into the ego vehicle frame for the
            # timestamp of the image.
            poserecord = self.nusc.get("ego_pose", cam["ego_pose_token"])
            pc.translate(-np.array(poserecord["translation"]))
            pc.rotate(Quaternion(poserecord["rotation"]).rotation_matrix.T)

            # Fourth step: transform from ego into the camera.
            cs_record = self.nusc.get(
                "calibrated_sensor", cam["calibrated_sensor_token"]
            )
            pc.translate(-np.array(cs_record["translation"]))
            pc.rotate(Quaternion(cs_record["rotation"]).rotation_matrix.T)

            # Fifth step: actually take a "picture" of the point cloud.
            # Grab the depths (camera frame z axis points away from the camera).
            depths = pc.points[2, :]

            # Take the actual picture
            # (matrix multiplication with camera-matrix + renormalization).
            points = view_points(
                pc.points[:3, :],
                np.array(cs_record["camera_intrinsic"]),
                normalize=True,
            )

            # Remove points that are either outside or behind the camera.
            # Also make sure points are at least 1m in front of the camera to avoid
            # seeing the lidar points on the camera
            # casing for non-keyframes which are slightly out of sync.
            points = points[:2].T
            mask = np.ones(depths.shape[0], dtype=bool)
            mask = np.logical_and(mask, depths > min_dist)
            mask = np.logical_and(mask, points[:, 0] > 0)
            mask = np.logical_and(mask, points[:, 0] < im.shape[1] - 1)
            mask = np.logical_and(mask, points[:, 1] > 0)
            mask = np.logical_and(mask, points[:, 1] < im.shape[0] - 1)
            matching_points = np.where(mask)[0]
            matching_pixels = np.round(
                np.flip(points[matching_points], axis=1)
            ).astype(np.int64)
            images.append(im / 255)
            pairing_points = np.concatenate((pairing_points, matching_points))
            pairing_images = np.concatenate(
                (
                    pairing_images,
                    np.concatenate(
                        (
                            np.ones((matching_pixels.shape[0], 1), dtype=np.int64) * i,
                            matching_pixels,
                        ),
                        axis=1,
                    ),
                )
            )
        return pc_ref.T, ts_pc_ref.T, images, pairing_points, pairing_images, np.stack(superpixels)

    def __len__(self):
        return len(self.list_keyframes)

    def unique(self, x, dim=-1):
        unique, inverse_org = torch.unique(x, return_inverse=True, dim=dim)
        perm = torch.arange(inverse_org.size(dim), dtype=inverse_org.dtype, device=inverse_org.device)
        inverse, perm = inverse_org.flip([dim]), perm.flip([dim])
        return unique, inverse.new_empty(unique.size(dim)).scatter_(dim, inverse, perm), inverse_org

    def __getitem__(self, idx):
        (
            pc,
            ts_pc,
            images,
            pairing_points,
            pairing_images,
            superpixels,
        ) = self.map_pointcloud_to_image(self.list_keyframes[idx])
        superpixels = torch.tensor(superpixels)

        # # get the temporal shift pc
        # pointsensor = self.nusc.get("sample_data", self.list_keyframes[idx]["LIDAR_TOP"])
        # current_sample = self.nusc.get("sample", pointsensor["sample_token"])
        # ts_sample_token = current_sample['next'] if current_sample['next'] != '' else current_sample['prev']
        # ts_sample = self.nusc.get("sample", ts_sample_token)
        # ts_data = ts_sample['data']
        # (
        #     ts_pc,
        #     _,
        #     pairing_points_ts,
        #     pairing_images_ts,
        #     superpixels_ts,
        # ) = self.map_pointcloud_to_image(ts_data)

        intensity = torch.tensor(pc[:, 3:])
        pc = torch.tensor(pc[:, :3])
        ts_intensity = torch.tensor(ts_pc[:, 3:])
        ts_pc = torch.tensor(ts_pc[:, :3])
        images = torch.tensor(np.array(images, dtype=np.float32).transpose(0, 3, 1, 2))

        if self.cloud_transforms:
            pc = self.cloud_transforms(pc)
            ts_pc = self.cloud_transforms(ts_pc)
        if self.mixed_transforms:
            (
                pc,
                intensity,
                images,
                pairing_points,
                pairing_images,
                superpixels,
            ) = self.mixed_transforms(
                pc, intensity, images, pairing_points, pairing_images, superpixels
            )

        if self.cloud_transforms:
            pc2 = self.cloud_transforms(copy.deepcopy(pc))
        else:
            pc2 = copy.deepcopy(pc)

        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            pc_size = pc.size(0)
            pc = torch.cat((pc, pc2, ts_pc), dim=0)
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            phi = torch.atan2(y, x) * 180 / np.pi  # corresponds to a split each 1°
            z = z / self.voxel_size
            coords_aug_triple = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
            coords_aug = coords_aug_triple[:pc_size]
            coords_aug_2 = coords_aug_triple[pc_size: 2*pc_size]
            ts_coords_aug = coords_aug_triple[2*pc_size:]
        else:
            coords_aug = pc / self.voxel_size
            coords_aug_2 = pc2 / self.voxel_size
            ts_coords_aug = ts_pc / self.voxel_size

        # Voxelization with MinkowskiEngine
        # for pc
        discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
            coords_aug.contiguous(), return_index=True, return_inverse=True
        )
        # indexes here are the indexes of points kept after the voxelization
        pairing_points = inverse_indexes[pairing_points]

        unique_feats = intensity[indexes]

        discrete_coords = torch.cat(
            (
                torch.zeros(discrete_coords.shape[0], 1, dtype=torch.int32),
                discrete_coords,
            ),
            1,
        )

        # for pc2
        discrete_coords2, indexes2, inverse_indexes2 = ME.utils.sparse_quantize(
            coords_aug_2.contiguous(), return_index=True, return_inverse=True
        )
        unique_feats2 = intensity[indexes]  # here adopt indexes, instead of indexes2
        discrete_coords2 = discrete_coords2[inverse_indexes2][indexes]
        discrete_coords2 = torch.cat(
            (
                torch.zeros(discrete_coords2.shape[0], 1, dtype=torch.int32),
                discrete_coords2,
            ),
            1,
        )

        # remove the repeat points in discrete_coords2 caused by remap to the order of coords1
        discrete_coords2, unique_index, unique_index_reverse = self.unique(discrete_coords2, dim=0)
        unique_feats2 = unique_feats2[unique_index]
        discrete_coords = discrete_coords[unique_index]
        unique_feats = unique_feats[unique_index]
        pairing_points = unique_index_reverse[pairing_points]
        inverse_indexes = unique_index_reverse[inverse_indexes]

        # for ts_pc
        ts_discrete_coords, ts_indexes, ts_inverse_indexes = ME.utils.sparse_quantize(
            ts_coords_aug.contiguous(), return_index=True, return_inverse=True
        )
        # pairing_points_ts = ts_inverse_indexes[pairing_points_ts]

        ts_unique_feats = ts_intensity[ts_indexes]

        ts_discrete_coords = torch.cat(
            (
                torch.zeros(ts_discrete_coords.shape[0], 1, dtype=torch.int32),
                ts_discrete_coords,
            ),
            1,
        )

        return (
            discrete_coords,
            unique_feats,
            discrete_coords2,
            unique_feats2,
            ts_discrete_coords,
            ts_unique_feats,
            images,
            pairing_points,
            # pairing_points_ts,
            pairing_images,
            # pairing_images_ts,
            inverse_indexes,
            superpixels,
            # superpixels_ts,
        )

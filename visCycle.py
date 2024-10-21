from mox_util.setup import setup

setup()

import os
import argparse
import torch
import numpy as np
import MinkowskiEngine as ME
from MinkowskiEngine import SparseTensor
from utils.read_config import generate_config
from downstream.model_builder import make_model
from nuscenes.nuscenes import NuScenes, NuScenesExplorer
from nuscenes.utils.data_classes import LidarPointCloud
from utils.transforms import (
    make_transforms_clouds,
)

import json
import math
import os
import os.path as osp
import sys
import time
from datetime import datetime
from typing import Tuple, List, Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics
from PIL import Image
from matplotlib import rcParams
from matplotlib.axes import Axes
from pyquaternion import Quaternion
from tqdm import tqdm

from nuscenes.lidarseg.lidarseg_utils import colormap_to_colors, plt_to_cv2, get_stats, \
    get_labels_in_coloring, create_lidarseg_legend, paint_points_label
from nuscenes.panoptic.panoptic_utils import paint_panop_points_label, stuff_cat_ids, get_frame_panoptic_instances,\
    get_panoptic_instances_stats
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from nuscenes.utils.data_io import load_bin_file, panoptic_to_lidarseg
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from nuscenes.utils.map_mask import MapMask
from nuscenes.utils.color_map import get_colormap


def main():
    """
    Code for launching the downstream training
    """
    parser = argparse.ArgumentParser(description="arg parser")
    parser.add_argument(
        "--cfg_file", type=str, default="config/semseg_nuscenes.yaml", help="specify the config for training"
    )
    parser.add_argument(
        "--pretraining_path", type=str, default=None, help="provide a path to pre-trained weights"
    )


    parser.add_argument(
        "--rank",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--world_size",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )


    args = parser.parse_args()
    config = generate_config(args.cfg_file)
    if args.pretraining_path:
        config['pretraining_path'] = args.pretraining_path

    if os.environ.get("LOCAL_RANK", 0) == 0:
        print(
            "\n" + "\n".join(list(map(lambda x: f"{x[0]:20}: {x[1]}", config.items())))
        )

    nusc, list_tokens = get_dataset(config)

    idx = 500
    print('forwarding...')
    if os.path.exists(os.path.join(config['working_dir'], 'matching_matrix_{}'.format(idx))):
        matching = torch.load(os.path.join(config['working_dir'], 'matching_matrix_{}'.format(idx)))
    else:
        model = make_model(config, config["pretraining_path"]).cuda()
        data = get_sample(nusc, list_tokens, idx, config)
        matching = forward(data, model)
        torch.save(matching, os.path.join(config['working_dir'], 'matching_matrix_{}'.format(idx)))

    print('vis...')
    vis(matching, nusc, list_tokens, idx, config)

def get_dataset(config):
    nusc = NuScenes_custom(
        version="v1.0-trainval", dataroot=config["data_root"], verbose=False
    )
    list_tokens = []
    for scene_idx in range(len(nusc.scene)):
        scene = nusc.scene[scene_idx]
        current_sample_token = scene["first_sample_token"]
        while current_sample_token != "":
            current_sample = nusc.get("sample", current_sample_token)
            next_sample_token = current_sample["next"]
            list_tokens.append(current_sample["data"])
            current_sample_token = next_sample_token

    return nusc, list_tokens


def get_sample(nusc, list_tokens, idx, config):
    sample = list_tokens[idx]
    pointsensor = nusc.get("sample_data", sample["LIDAR_TOP"])
    pcl_path = os.path.join(nusc.dataroot, pointsensor["filename"])
    points = LidarPointCloud.from_file(pcl_path).points.T
    # get the points (4th coordinate is the point intensity)
    pc = points[:, :3]

    current_sample = nusc.get("sample", pointsensor["sample_token"])
    ts_sample_token = current_sample['next'] if current_sample['next'] != '' else current_sample['prev']
    ts_sample = nusc.get("sample", ts_sample_token)
    ts_data = ts_sample['data']
    ts_pointsensor = nusc.get("sample_data", ts_data["LIDAR_TOP"])
    ts_pcl_path = os.path.join(nusc.dataroot, ts_pointsensor["filename"])
    ts_points = LidarPointCloud.from_file(ts_pcl_path).points.T
    ts_pc = ts_points[:, :3]

    pc = torch.tensor(pc)
    ts_pc = torch.tensor(ts_pc)

    # apply the transforms (augmentation)
    cloud_transforms = make_transforms_clouds(config)
    pc = cloud_transforms(pc)
    ts_pc = cloud_transforms(ts_pc)

    # Transform to cylinder coordinate and scale for given voxel size
    pc_size = pc.size(0)
    pc = torch.cat((pc, ts_pc), dim=0)
    x, y, z = pc.T
    rho = torch.sqrt(x ** 2 + y ** 2) / 0.1
    # corresponds to a split each 1°
    phi = torch.atan2(y, x) * 180 / np.pi
    z = z / 0.1
    coords_aug_both = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
    coords_aug = coords_aug_both[:pc_size]
    ts_coords_aug = coords_aug_both[pc_size:]

    # Voxelization
    discrete_coords, indexes, inverse_indexes = ME.utils.sparse_quantize(
        coords_aug, return_index=True, return_inverse=True
    )
    ts_discrete_coords, ts_indexes, ts_inverse_indexes = ME.utils.sparse_quantize(
        ts_coords_aug.contiguous(), return_index=True, return_inverse=True
    )

    # use those voxels features
    unique_feats = torch.tensor(points[indexes][:, 3:])
    ts_unique_feats = torch.tensor(ts_points[ts_indexes][:, 3:])



    coords_batch, len_batch, ts_coords_batch, ts_len_batch = [], [], [], []

    # create a tensor of coordinates of the 3D points
    # note that in ME, batche index and point indexes are collated in the same dimension
    N = discrete_coords.shape[0]
    coords_batch.append(
        torch.cat((torch.ones(N, 1, dtype=torch.int32) * 0, discrete_coords), 1)
    )
    len_batch.append(N)

    N = ts_discrete_coords.shape[0]
    ts_coords_batch.append(
        torch.cat((torch.ones(N, 1, dtype=torch.int32) * 0, ts_discrete_coords), 1)
    )
    ts_len_batch.append(N)

    # Collate all lists on their first dimension
    coords_batch = torch.cat(coords_batch, 0).int()
    feats_batch = unique_feats.float()
    ts_coords_batch = torch.cat(ts_coords_batch, 0).int()
    ts_feats_batch = ts_unique_feats.float()
    return {
        "pc": pc,
        'ts_pc': ts_pc,
        "sinput_C": coords_batch,
        "ts_sinput_C": ts_coords_batch,
        "sinput_F": feats_batch,
        "ts_sinput_F": ts_feats_batch,
        "len_batch": len_batch,
        "ts_len_batch": ts_len_batch,
        "inverse_indexes": inverse_indexes,
        "ts_inverse_indexes": ts_inverse_indexes,
    }


def forward(batch, model):
    model.eval()
    with torch.no_grad():
        sparse_input = SparseTensor(batch["sinput_F"], batch["sinput_C"], device='cuda')
        output_points, output_points_cycle = model(sparse_input)
        ts_sparse_input = SparseTensor(batch["ts_sinput_F"], batch["ts_sinput_C"], device='cuda')
        _, ts_output_points_cycle = model(ts_sparse_input)
        matching = torch.mm(output_points_cycle.F, ts_output_points_cycle.F.T) # n, m
        matching = matching[:, batch['ts_inverse_indexes']]
        matching = matching[batch['inverse_indexes']]

        # rand_idx = np.random.choice(output_points_cycle.F.shape[0], 2048, replace=False)
        # output_points_cycle_selected = output_points_cycle.F[rand_idx]
        # A12 = torch.nn.functional.softmax(torch.mm(output_points_cycle_selected, ts_output_points_cycle.F.T) / 0.1, dim=-1)
        # A21 = torch.nn.functional.softmax(torch.mm(ts_output_points_cycle.F, output_points_cycle_selected.T) / 0.1, dim=-1)
        # jump_matrix = torch.mm(A12, A21)
        # print(jump_matrix[:10, :10])
        # print(jump_matrix.sort(dim=1, descending=True)[0][:, :10])
        # print(jump_matrix.argmax(dim=1)[:10])

    return matching

@torch.no_grad()
def vis(matching, nusc, list_tokens, idx, config):
    matching_org = matching
    for i in range(0, len(matching), 1000):
        input('Press Enter to continue...')
        print('vis....{}'.format(i))
        pixel_idx = i
        matching = matching_org[pixel_idx]
        matching = torch.clamp((matching + 1) / 2, min=0, max=0.999)
        matching = torch.floor(matching * 32)
        matching_point = torch.argmax(matching)

        matching = matching.data.cpu().numpy().astype(np.uint8)
        matching.tofile(os.path.join(config['working_dir'], 'binfile'))

        sample_data = list_tokens[idx]
        pointsensor = nusc.get("sample_data", sample_data["LIDAR_TOP"])
        current_sample = nusc.get("sample", pointsensor["sample_token"])
        ts_sample_token = current_sample['next'] if current_sample['next'] != '' else current_sample['prev']
        ts_sample = nusc.get("sample", ts_sample_token)

        nusc.render_matching(ts_sample['data']['LIDAR_TOP'], out_path=os.path.join(config['working_dir'], '{}_vis_ts'.format(idx)), underlay_map=False,
                                with_anns=False, lidarseg_preds_bin_path=os.path.join(config['working_dir'], 'binfile'),
                                use_flat_vehicle_coordinates=False, show_lidarseg=True, show_lidarseg_legend=True, point_coord=matching_point)

        nusc.render_lidar_point(current_sample['data']['LIDAR_TOP'], out_path=os.path.join(config['working_dir'], '{}_vis_current'.format(idx)), underlay_map=False,
                                with_anns=False, use_flat_vehicle_coordinates=False, show_lidarseg=True, show_lidarseg_legend=True, point_coord=pixel_idx)


class NuScenes_custom(NuScenes):
    def __init__(self,
                 version: str = 'v1.0-mini',
                 dataroot: str = '/data/sets/nuscenes',
                 verbose: bool = True,
                 map_resolution: float = 0.1):
        super().__init__(version, dataroot, verbose, map_resolution)
        self.colormap_rainbow = get_colormap()
        self.colormap_rainbow = dict({c['name']: self.colormap_rainbow[c['name']]
                              for c in sorted(self.category, key=lambda k: k['index'])})
        self.explorer = NuScenesExplorsr_custom(self)

    def render_matching(self, sample_data_token: str, with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY, axes_limit: float = 40, ax: Axes = None,
                           nsweeps: int = 1, out_path: str = None, underlay_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           show_lidarseg: bool = False,
                           show_lidarseg_legend: bool = False,
                           filter_lidarseg_labels: List = None,
                           lidarseg_preds_bin_path: str = None, verbose: bool = True,
                           show_panoptic: bool = False,
                        point_coord=None) -> None:
        self.explorer.render_matching(sample_data_token, with_anns, box_vis_level, axes_limit, ax, nsweeps=nsweeps,
                                         out_path=out_path,
                                         underlay_map=underlay_map,
                                         use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                                         show_lidarseg=show_lidarseg,
                                         show_lidarseg_legend=show_lidarseg_legend,
                                         filter_lidarseg_labels=filter_lidarseg_labels,
                                         lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                         verbose=verbose,
                                         show_panoptic=show_panoptic,
                                      point_coord=point_coord)

    def render_lidar_point(self, sample_data_token: str, with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY, axes_limit: float = 40,
                           ax: Axes = None,
                           nsweeps: int = 1, out_path: str = None, underlay_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           show_lidarseg: bool = False,
                           show_lidarseg_legend: bool = False,
                           filter_lidarseg_labels: List = None,
                           lidarseg_preds_bin_path: str = None, verbose: bool = True,
                           show_panoptic: bool = False,
                           point_coord=None) -> None:
        self.explorer.render_lidar_point(sample_data_token, with_anns, box_vis_level, axes_limit, ax,
                                         nsweeps=nsweeps,
                                         out_path=out_path,
                                         underlay_map=underlay_map,
                                         use_flat_vehicle_coordinates=use_flat_vehicle_coordinates,
                                         show_lidarseg=show_lidarseg,
                                         show_lidarseg_legend=show_lidarseg_legend,
                                         filter_lidarseg_labels=filter_lidarseg_labels,
                                         lidarseg_preds_bin_path=lidarseg_preds_bin_path,
                                         verbose=verbose,
                                         show_panoptic=show_panoptic,
                                         show_point_coord=point_coord)

class NuScenesExplorsr_custom(NuScenesExplorer):
    def render_lidar_point(self,
                           sample_data_token: str,
                           with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: float = 40,
                           ax: Axes = None,
                           nsweeps: int = 1,
                           out_path: str = None,
                           underlay_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           show_lidarseg: bool = False,
                           show_lidarseg_legend: bool = False,
                           filter_lidarseg_labels: List = None,
                           lidarseg_preds_bin_path: str = None,
                           verbose: bool = True,
                           show_panoptic: bool = False,
                           show_point_coord=None) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw box annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to display the image after it is rendered.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        if show_lidarseg:
            show_panoptic = False
        # Get sensor modality.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            sample_rec = self.nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

            if sensor_modality == 'lidar':
                if show_lidarseg or show_panoptic:
                    gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                    assert hasattr(self.nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'

                    # Ensure that lidar pointcloud is from a keyframe.
                    assert sd_record['is_key_frame'], \
                        'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                    assert nsweeps == 1, \
                        'Error: Only pointclouds which are keyframes have lidar segmentation labels; nsweeps should ' \
                        'be set to 1.'

                    # Load a single lidar point cloud.
                    pcl_path = osp.join(self.nusc.dataroot, ref_sd_record['filename'])
                    pc = LidarPointCloud.from_file(pcl_path)
                else:
                    # Get aggregated lidar point cloud in lidar frame.
                    pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan,
                                                                     nsweeps=nsweeps)
                velocities = None
            else:
                # Get aggregated radar point cloud in reference frame.
                # The point cloud is transformed to the reference frame for visualization purposes.
                pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

                # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
                # point cloud.
                radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                velocities = pc.points[8:10, :]  # Compensated velocity
                velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
                velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
                velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
                velocities[2, :] = np.zeros(pc.points.shape[1])

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Render map if requested.
            if underlay_map:
                assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                     'otherwise the location does not correspond to the map!'
                self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            if sensor_modality == 'lidar' and (show_lidarseg or show_panoptic):
                gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                semantic_table = getattr(self.nusc, gt_from)
                # Load labels for pointcloud.
                if lidarseg_preds_bin_path:
                    sample_token = self.nusc.get('sample_data', sample_data_token)['sample_token']
                    lidarseg_labels_filename = lidarseg_preds_bin_path
                    assert os.path.exists(lidarseg_labels_filename), \
                        'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                        'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, sample_data_token)
                else:
                    if len(semantic_table) > 0:
                        # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                        lidarseg_labels_filename = osp.join(self.nusc.dataroot,
                                                            self.nusc.get(gt_from, sample_data_token)['filename'])
                    else:
                        lidarseg_labels_filename = None

                if lidarseg_labels_filename:
                    # Paint each label in the pointcloud with a RGBA value.
                    if show_lidarseg or show_panoptic:
                        if show_lidarseg:
                            colors = paint_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                        self.nusc.lidarseg_name2idx_mapping, self.nusc.colormap)
                        else:
                            colors = paint_panop_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                              self.nusc.lidarseg_name2idx_mapping, self.nusc.colormap)

                        if show_lidarseg_legend:

                            # If user does not specify a filter, then set the filter to contain the classes present in
                            # the pointcloud after it has been projected onto the image; this will allow displaying the
                            # legend only for classes which are present in the image (instead of all the classes).
                            if filter_lidarseg_labels is None:
                                if show_lidarseg:
                                    # Since the labels are stored as class indices, we get the RGB colors from the
                                    # colormap in an array where the position of the RGB color corresponds to the index
                                    # of the class it represents.
                                    color_legend = colormap_to_colors(self.nusc.colormap,
                                                                      self.nusc.lidarseg_name2idx_mapping)
                                    filter_lidarseg_labels = get_labels_in_coloring(color_legend, colors)
                                else:
                                    # Only show legends for stuff categories for panoptic.
                                    filter_lidarseg_labels = stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping))

                            if filter_lidarseg_labels and show_panoptic:
                                # Only show legends for filtered stuff categories for panoptic.
                                stuff_labels = set(stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping)))
                                filter_lidarseg_labels = list(stuff_labels.intersection(set(filter_lidarseg_labels)))

                            create_lidarseg_legend(filter_lidarseg_labels,
                                                   self.nusc.lidarseg_idx2name_mapping,
                                                   self.nusc.colormap,
                                                   loc='upper left',
                                                   ncol=1,
                                                   bbox_to_anchor=(1.05, 1.0))
                else:
                    print('Warning: There are no lidarseg labels in {}. Points will be colored according to distance '
                          'from the ego vehicle instead.'.format(self.nusc.version))

            point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

            # Show velocities.
            if sensor_modality == 'radar':
                points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
                deltas_vel = points_vel - points
                deltas_vel = 6 * deltas_vel  # Arbitrary scaling
                max_delta = 20
                deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
                colors_rgba = scatter.to_rgba(colors)
                for i in range(points.shape[1]):
                    ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')

            if show_point_coord is not None:
                ax.plot(points[0, show_point_coord], points[1, show_point_coord], 'x', color='red')

            # Get boxes in lidar frame.
            _, boxes, _ = self.nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                    use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)
        elif sensor_modality == 'camera':
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_data_token,
                                                                           box_vis_level=box_vis_level)
            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax.axis('off')
        ax.set_title('{} {labels_type}'.format(
            sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        ax.set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

        if verbose:
            plt.show()

    def render_matching(self,
                           sample_data_token: str,
                           with_anns: bool = True,
                           box_vis_level: BoxVisibility = BoxVisibility.ANY,
                           axes_limit: float = 40,
                           ax: Axes = None,
                           nsweeps: int = 1,
                           out_path: str = None,
                           underlay_map: bool = True,
                           use_flat_vehicle_coordinates: bool = True,
                           show_lidarseg: bool = False,
                           show_lidarseg_legend: bool = False,
                           filter_lidarseg_labels: List = None,
                           lidarseg_preds_bin_path: str = None,
                           verbose: bool = True,
                           show_panoptic: bool = False,
                        point_coord=None) -> None:
        """
        Render sample data onto axis.
        :param sample_data_token: Sample_data token.
        :param with_anns: Whether to draw box annotations.
        :param box_vis_level: If sample_data is an image, this sets required visibility for boxes.
        :param axes_limit: Axes limit for lidar and radar (measured in meters).
        :param ax: Axes onto which to render.
        :param nsweeps: Number of sweeps for lidar and radar.
        :param out_path: Optional path to save the rendered figure to disk.
        :param underlay_map: When set to true, lidar data is plotted onto the map. This can be slow.
        :param use_flat_vehicle_coordinates: Instead of the current sensor's coordinate frame, use ego frame which is
            aligned to z-plane in the world. Note: Previously this method did not use flat vehicle coordinates, which
            can lead to small errors when the vertical axis of the global frame and lidar are not aligned. The new
            setting is more correct and rotates the plot by ~90 degrees.
        :param show_lidarseg: When set to True, the lidar data is colored with the segmentation labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
        :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
        :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
            or the list is empty, all classes will be displayed.
        :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
                                        predictions for the sample.
        :param verbose: Whether to display the image after it is rendered.
        :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
            to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
            If show_lidarseg is True, show_panoptic will be set to False.
        """
        if show_lidarseg:
            show_panoptic = False
        # Get sensor modality.
        sd_record = self.nusc.get('sample_data', sample_data_token)
        sensor_modality = sd_record['sensor_modality']

        if sensor_modality in ['lidar', 'radar']:
            sample_rec = self.nusc.get('sample', sd_record['sample_token'])
            chan = sd_record['channel']
            ref_chan = 'LIDAR_TOP'
            ref_sd_token = sample_rec['data'][ref_chan]
            ref_sd_record = self.nusc.get('sample_data', ref_sd_token)

            if sensor_modality == 'lidar':
                if show_lidarseg or show_panoptic:
                    gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                    assert hasattr(self.nusc, gt_from), f'Error: nuScenes-{gt_from} not installed!'

                    # Ensure that lidar pointcloud is from a keyframe.
                    assert sd_record['is_key_frame'], \
                        'Error: Only pointclouds which are keyframes have lidar segmentation labels. Rendering aborted.'

                    assert nsweeps == 1, \
                        'Error: Only pointclouds which are keyframes have lidar segmentation labels; nsweeps should ' \
                        'be set to 1.'

                    # Load a single lidar point cloud.
                    pcl_path = osp.join(self.nusc.dataroot, ref_sd_record['filename'])
                    pc = LidarPointCloud.from_file(pcl_path)
                else:
                    # Get aggregated lidar point cloud in lidar frame.
                    pc, times = LidarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan,
                                                                     nsweeps=nsweeps)
                velocities = None
            else:
                # Get aggregated radar point cloud in reference frame.
                # The point cloud is transformed to the reference frame for visualization purposes.
                pc, times = RadarPointCloud.from_file_multisweep(self.nusc, sample_rec, chan, ref_chan, nsweeps=nsweeps)

                # Transform radar velocities (x is front, y is left), as these are not transformed when loading the
                # point cloud.
                radar_cs_record = self.nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
                ref_cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                velocities = pc.points[8:10, :]  # Compensated velocity
                velocities = np.vstack((velocities, np.zeros(pc.points.shape[1])))
                velocities = np.dot(Quaternion(radar_cs_record['rotation']).rotation_matrix, velocities)
                velocities = np.dot(Quaternion(ref_cs_record['rotation']).rotation_matrix.T, velocities)
                velocities[2, :] = np.zeros(pc.points.shape[1])

            # By default we render the sample_data top down in the sensor frame.
            # This is slightly inaccurate when rendering the map as the sensor frame may not be perfectly upright.
            # Using use_flat_vehicle_coordinates we can render the map in the ego frame instead.
            if use_flat_vehicle_coordinates:
                # Retrieve transformation matrices for reference point cloud.
                cs_record = self.nusc.get('calibrated_sensor', ref_sd_record['calibrated_sensor_token'])
                pose_record = self.nusc.get('ego_pose', ref_sd_record['ego_pose_token'])
                ref_to_ego = transform_matrix(translation=cs_record['translation'],
                                              rotation=Quaternion(cs_record["rotation"]))

                # Compute rotation between 3D vehicle pose and "flat" vehicle pose (parallel to global z plane).
                ego_yaw = Quaternion(pose_record['rotation']).yaw_pitch_roll[0]
                rotation_vehicle_flat_from_vehicle = np.dot(
                    Quaternion(scalar=np.cos(ego_yaw / 2), vector=[0, 0, np.sin(ego_yaw / 2)]).rotation_matrix,
                    Quaternion(pose_record['rotation']).inverse.rotation_matrix)
                vehicle_flat_from_vehicle = np.eye(4)
                vehicle_flat_from_vehicle[:3, :3] = rotation_vehicle_flat_from_vehicle
                viewpoint = np.dot(vehicle_flat_from_vehicle, ref_to_ego)
            else:
                viewpoint = np.eye(4)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 9))

            # Render map if requested.
            if underlay_map:
                assert use_flat_vehicle_coordinates, 'Error: underlay_map requires use_flat_vehicle_coordinates, as ' \
                                                     'otherwise the location does not correspond to the map!'
                self.render_ego_centric_map(sample_data_token=sample_data_token, axes_limit=axes_limit, ax=ax)

            # Show point cloud.
            points = view_points(pc.points[:3, :], viewpoint, normalize=False)
            dists = np.sqrt(np.sum(pc.points[:2, :] ** 2, axis=0))
            colors = np.minimum(1, dists / axes_limit / np.sqrt(2))
            if sensor_modality == 'lidar' and (show_lidarseg or show_panoptic):
                gt_from = 'lidarseg' if show_lidarseg else 'panoptic'
                semantic_table = getattr(self.nusc, gt_from)
                # Load labels for pointcloud.
                if lidarseg_preds_bin_path:
                    sample_token = self.nusc.get('sample_data', sample_data_token)['sample_token']
                    lidarseg_labels_filename = lidarseg_preds_bin_path
                    assert os.path.exists(lidarseg_labels_filename), \
                        'Error: Unable to find {} to load the predictions for sample token {} (lidar ' \
                        'sample data token {}) from.'.format(lidarseg_labels_filename, sample_token, sample_data_token)
                else:
                    if len(semantic_table) > 0:
                        # Ensure {lidarseg/panoptic}.json is not empty (e.g. in case of v1.0-test).
                        lidarseg_labels_filename = osp.join(self.nusc.dataroot,
                                                            self.nusc.get(gt_from, sample_data_token)['filename'])
                    else:
                        lidarseg_labels_filename = None

                if lidarseg_labels_filename:
                    # Paint each label in the pointcloud with a RGBA value.
                    if show_lidarseg or show_panoptic:
                        if show_lidarseg:
                            colors = paint_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                        self.nusc.lidarseg_name2idx_mapping, self.nusc.colormap_rainbow)
                        else:
                            colors = paint_panop_points_label(lidarseg_labels_filename, filter_lidarseg_labels,
                                                              self.nusc.lidarseg_name2idx_mapping, self.nusc.colormap_rainbow)

                        if show_lidarseg_legend:

                            # If user does not specify a filter, then set the filter to contain the classes present in
                            # the pointcloud after it has been projected onto the image; this will allow displaying the
                            # legend only for classes which are present in the image (instead of all the classes).
                            if filter_lidarseg_labels is None:
                                if show_lidarseg:
                                    # Since the labels are stored as class indices, we get the RGB colors from the
                                    # colormap in an array where the position of the RGB color corresponds to the index
                                    # of the class it represents.
                                    color_legend = colormap_to_colors(self.nusc.colormap_rainbow,
                                                                      self.nusc.lidarseg_name2idx_mapping)
                                    filter_lidarseg_labels = get_labels_in_coloring(color_legend, colors)
                                else:
                                    # Only show legends for stuff categories for panoptic.
                                    filter_lidarseg_labels = stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping))

                            if filter_lidarseg_labels and show_panoptic:
                                # Only show legends for filtered stuff categories for panoptic.
                                stuff_labels = set(stuff_cat_ids(len(self.nusc.lidarseg_name2idx_mapping)))
                                filter_lidarseg_labels = list(stuff_labels.intersection(set(filter_lidarseg_labels)))

                            create_lidarseg_legend(filter_lidarseg_labels,
                                                   self.nusc.lidarseg_idx2name_mapping,
                                                   self.nusc.colormap_rainbow,
                                                   loc='upper left',
                                                   ncol=1,
                                                   bbox_to_anchor=(1.05, 1.0))
                else:
                    print('Warning: There are no lidarseg labels in {}. Points will be colored according to distance '
                          'from the ego vehicle instead.'.format(self.nusc.version))

            point_scale = 0.2 if sensor_modality == 'lidar' else 3.0
            scatter = ax.scatter(points[0, :], points[1, :], c=colors, s=point_scale)

            # Show velocities.
            if sensor_modality == 'radar':
                points_vel = view_points(pc.points[:3, :] + velocities, viewpoint, normalize=False)
                deltas_vel = points_vel - points
                deltas_vel = 6 * deltas_vel  # Arbitrary scaling
                max_delta = 20
                deltas_vel = np.clip(deltas_vel, -max_delta, max_delta)  # Arbitrary clipping
                colors_rgba = scatter.to_rgba(colors)
                for i in range(points.shape[1]):
                    ax.arrow(points[0, i], points[1, i], deltas_vel[0, i], deltas_vel[1, i], color=colors_rgba[i])

            # Show ego vehicle.
            ax.plot(0, 0, 'x', color='red')
            if point_coord is not None:
                ax.plot(points[0, point_coord], points[1, point_coord], 'x', color='red')

            # Get boxes in lidar frame.
            _, boxes, _ = self.nusc.get_sample_data(ref_sd_token, box_vis_level=box_vis_level,
                                                    use_flat_vehicle_coordinates=use_flat_vehicle_coordinates)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=np.eye(4), colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(-axes_limit, axes_limit)
            ax.set_ylim(-axes_limit, axes_limit)
        elif sensor_modality == 'camera':
            # Load boxes and image.
            data_path, boxes, camera_intrinsic = self.nusc.get_sample_data(sample_data_token,
                                                                           box_vis_level=box_vis_level)
            data = Image.open(data_path)

            # Init axes.
            if ax is None:
                _, ax = plt.subplots(1, 1, figsize=(9, 16))

            # Show image.
            ax.imshow(data)

            # Show boxes.
            if with_anns:
                for box in boxes:
                    c = np.array(self.get_color(box.name)) / 255.0
                    box.render(ax, view=camera_intrinsic, normalize=True, colors=(c, c, c))

            # Limit visible range.
            ax.set_xlim(0, data.size[0])
            ax.set_ylim(data.size[1], 0)

        else:
            raise ValueError("Error: Unknown sensor modality!")

        ax.axis('off')
        ax.set_title('{} {labels_type}'.format(
            sd_record['channel'], labels_type='(predictions)' if lidarseg_preds_bin_path else ''))
        ax.set_aspect('equal')

        if out_path is not None:
            plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)

        if verbose:
            plt.show()

def get_colormap():
    """
    Get the defined colormap.
    :return: A mapping from the class names to the respective RGB values.
    """

    classname_to_color = {  # RGB.
        "noise": (0, 0, 122),  # Black.
        "animal": (0, 0, 164),  # Steelblue
        "human.pedestrian.adult": (0, 0, 202),  # Blue
        "human.pedestrian.child": (0, 0, 237),  # Skyblue,
        "human.pedestrian.construction_worker": (0, 0, 255),  # Cornflowerblue
        "human.pedestrian.personal_mobility": (0, 36, 255),  # Palevioletred
        "human.pedestrian.police_officer": (0, 68, 255),  # Navy,
        "human.pedestrian.stroller": (0, 100, 255),  # Lightcoral
        "human.pedestrian.wheelchair": (0, 136, 255),  # Blueviolet
        "movable_object.barrier": (0, 168, 255),  # Slategrey
        "movable_object.debris": (0, 200, 255),  # Chocolate
        "movable_object.pushable_pullable": (3, 234, 243),  # Dimgrey
        "movable_object.trafficcone": (31, 255, 215),  # Darkslategrey
        "static_object.bicycle_rack": (57, 255, 190),  # Rosybrown
        "vehicle.bicycle": (83, 255, 164),  # Crimson
        "vehicle.bus.bendy": (109, 255, 138),  # Coral
        "vehicle.bus.rigid": (138, 255, 109),  # Orangered
        "vehicle.car": (164, 255, 83),  # Orange
        "vehicle.construction": (190, 255, 60),  # Darksalmon
        "vehicle.emergency.ambulance": (219, 255, 28),
        "vehicle.emergency.police": (243, 249, 3),  # Gold
        "vehicle.motorcycle": (255, 220, 0),  # Red
        "vehicle.trailer": (255, 190, 0),  # Darkorange
        "vehicle.truck": (255, 160, 0),  # Tomato
        "flat.driveable_surface": (255, 127, 0),  # nuTonomy green
        "flat.other": (255, 96, 0),
        "flat.sidewalk": (255, 67, 0),
        "flat.terrain": (255, 37, 0),
        "static.manmade": (241, 8, 0),  # Burlywood
        "static.other": (204, 0, 0),  # Bisque
        "static.vegetation": (165, 0, 0),  # Green
        "vehicle.ego": (131, 0, 0)
    }

    # classname_to_color = {  # RGB.
    #     "noise": (0, 0, 122, 0),  # Black.
    #     "animal": (0, 0, 164, 0),  # Steelblue
    #     "human.pedestrian.adult": (0, 0, 202, 0),  # Blue
    #     "human.pedestrian.child": (0, 0, 237, 0),  # Skyblue,
    #     "human.pedestrian.construction_worker": (0, 0, 255, 0),  # Cornflowerblue
    #     "human.pedestrian.personal_mobility": (0, 36, 255, 0),  # Palevioletred
    #     "human.pedestrian.police_officer": (0, 68, 255, 0),  # Navy,
    #     "human.pedestrian.stroller": (0, 100, 255, 0),  # Lightcoral
    #     "human.pedestrian.wheelchair": (0, 136, 255, 0),  # Blueviolet
    #     "movable_object.barrier": (0, 168, 255, 0),  # Slategrey
    #     "movable_object.debris": (0, 200, 255, 0),  # Chocolate
    #     "movable_object.pushable_pullable": (3, 234, 243, 0),  # Dimgrey
    #     "movable_object.trafficcone": (31, 255, 215, 0),  # Darkslategrey
    #     "static_object.bicycle_rack": (57, 255, 190, 0),  # Rosybrown
    #     "vehicle.bicycle": (83, 255, 164, 0),  # Crimson
    #     "vehicle.bus.bendy": (109, 255, 138, 0),  # Coral
    #     "vehicle.bus.rigid": (138, 255, 109, 0),  # Orangered
    #     "vehicle.car": (164, 255, 83, 0),  # Orange
    #     "vehicle.construction": (190, 255, 60, 0),  # Darksalmon
    #     "vehicle.emergency.ambulance": (219, 255, 28, 0),
    #     "vehicle.emergency.police": (243, 249, 3, 0),  # Gold
    #     "vehicle.motorcycle": (255, 220, 0, 0),  # Red
    #     "vehicle.trailer": (255, 190, 0, 0),  # Darkorange
    #     "vehicle.truck": (255, 160, 0, 255),  # Tomato
    #     "flat.driveable_surface": (255, 127, 0, 255),  # nuTonomy green
    #     "flat.other": (255, 96, 0, 255),
    #     "flat.sidewalk": (255, 67, 0, 255),
    #     "flat.terrain": (255, 37, 0, 255),
    #     "static.manmade": (241, 8, 0, 255),  # Burlywood
    #     "static.other": (204, 0, 0, 255),  # Bisque
    #     "static.vegetation": (165, 0, 0, 255),  # Green
    #     "vehicle.ego": (131, 0, 0, 255)
    # }

    return classname_to_color


if __name__ == "__main__":
    main()

from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version='v1.0-mini', dataroot='/home/zy/nuScenes', verbose=True)

my_sample = nusc.sample[20]
print(nusc.lidarseg_name2idx_mapping)
print(nusc.colormap)
print(nusc.lidarseg)
nusc.render_sample_data(my_sample['data']['LIDAR_TOP'], out_path = '/home/pb/slidr/vis/test.jpg', underlay_map=False, with_anns=False,
                        use_flat_vehicle_coordinates=False, show_lidarseg=True, show_lidarseg_legend=True)
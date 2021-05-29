from torch.utils import data
import numpy as np
import imageio

import os

from utils import (get_camera_rays, depths_to_world_coords, importance_sample_empty_points,
    frustum_cull, zs_to_depths)


class MultiObject3dDataset(data.Dataset):
    def __init__(self, path, mode, dataset='clevr3d', max_n=6, max_views=None,
                 points_per_item=2048, do_frustum_culling=False, max_len=None):
        """
        A PyTorch dataset for MultiObject3d data which operates as follows: Each instance consists
        of a single full view of a scene, which may be used as input to an encoder network. It also
        consists of a number of rays/pixels sampled from all available views of that scene, which
        may be used as supervision for training.
        Args:
            path: Directory of the dataset
            mode: 'train', 'val', or 'test'
            dataset: Name of the dataset, either 'clevr3d' or 'multishapenet'
            max_n: Only select scenes with at most this many objects
            max_views: Number of views to consider per scene
            points_per_item: Number of training rays to sample per instance
            do_frustum_culling: Only sample points in the frustum of the input view
            max_len: Limit the size of the dataset to this many scenes.
        """
        self.path = path
        self.mode = mode
        self.max_n = max_n
        self.points_per_item = points_per_item
        self.do_frustum_culling = do_frustum_culling
        if not dataset in {'clevr3d', 'multishapenet'}:
            raise ValueError(f'Unknown dataset: {dataset}')
        self.dataset = dataset
        self.max_len = max_len

        if self.dataset == 'multishapenet':
            self.max_num_entities = 5
            self.start_idx, self.end_idx = {'train': (0, 80000),
                                            'val': (80000, 80500),
                                            'test': (90000, 100000)}[mode]
        else:
            self.max_num_entities = 11
            self.start_idx, self.end_idx = {'train': (0, 70000),
                                            'val': (70000, 72000),
                                            'test': (85000, 100000)}[mode]

        self.metadata = np.load(os.path.join(path, 'metadata.npz'))
        self.metadata = {k: v for k, v in self.metadata.items()}

        num_objs = (self.metadata['shape'][self.start_idx:self.end_idx] > 0).sum(1)
        num_available_views = self.metadata['camera_pos'].shape[1]
        if max_views is None:
            self.num_views = num_available_views
        else:
            assert(self.max_views <= num_available_views)
            self.num_views = max_views

        self.idxs = np.arange(self.start_idx, self.end_idx)[num_objs <= max_n]

        print(f'Initialized {dataset} {mode} set, {len(self.idxs)} examples')

    def __len__(self):
        if self.max_len is not None:
            return self.max_len
        return len(self.idxs) * self.num_views

    def __getitem__(self, idx, noisy=True):
        scene_idx = idx % len(self.idxs)
        view_idx = idx // len(self.idxs)

        scene_idx = self.idxs[scene_idx]

        imgs = [np.asarray(imageio.imread(
            os.path.join(self.path, 'images', f'img_{scene_idx}_{v}.png')))
            for v in range(self.num_views)]
        depths = [np.asarray(imageio.imread(
            os.path.join(self.path, 'depths', f'depths_{scene_idx}_{v}.png')))
            for v in range(self.num_views)]

        imgs = [img[..., :3].astype(np.float32) / 255 for img in imgs]
        # Convert 16 bit integer depths to floating point numbers.
        # 0.025 is the normalization factor used while drawing the depthmaps.
        depths = [d.astype(np.float32) / (65536 * 0.025) for d in depths]

        input_img = np.transpose(imgs[view_idx], (2, 0, 1))

        metadata = {k: v[scene_idx] for (k, v) in self.metadata.items()}

        camera_pos = metadata['camera_pos'][view_idx]

        all_rays = []
        all_camera_pos = metadata['camera_pos']
        for i in range(self.num_views):
            cur_rays = get_camera_rays(all_camera_pos[i], noisy=False)
            all_rays.append(cur_rays)
        all_rays = np.stack(all_rays, 0)

        example = dict(metadata)

        if self.dataset == 'multishapenet':
            # We're not loading the path to the model files into PyTorch, since those are strings.
            del example['shape_file']

            # For the shapenet dataset, the depth images represent the z-coordinate in camera space.
            # Here, we convert this into Euclidian depths.
            new_depths = []
            for i in range(self.num_views):
                new_depth = zs_to_depths(depths[i], all_rays[i], all_camera_pos[i])
                new_depths.append(new_depth)
            depths = np.stack(new_depths, 0)

        example['view_idxs'] = view_idx  # The index of the view we're using as encoder input
        example['camera_pos'] = camera_pos.astype(np.float32)
        example['inputs'] = input_img  # The view we're using as encoder input
        # The direction of the rays of the input view
        example['input_rays'] = all_rays[view_idx].astype(np.float32)
        if self.mode != 'train':
            example['input_depths'] = depths[view_idx]  # The depths of the input view
        # The positions of the input view pixels in 3d world coordinates
        example['input_points'] = depths_to_world_coords(depths[view_idx],
                                                         example['input_rays'],
                                                         example['camera_pos'])


        all_values = np.reshape(np.stack(imgs, 0), (self.num_views * 240 * 320, 3))
        all_depths = np.reshape(np.stack(depths, 0), (self.num_views * 240 * 320,))
        all_rays = np.reshape(all_rays, (self.num_views * 240 * 320, 3))
        all_camera_pos = np.tile(np.expand_dims(all_camera_pos, 1), (1, 240 * 320, 1))
        all_camera_pos = np.reshape(all_camera_pos, (self.num_views * 240 * 320, 3))

        num_points = all_rays.shape[0]

        # If we have fewer points than we want, sample with replacement
        replace = num_points < self.points_per_item
        sampled_idxs = np.random.choice(np.arange(num_points),
                                        size=(self.points_per_item,),
                                        replace=replace)

        rays = all_rays[sampled_idxs]
        camera_pos = all_camera_pos[sampled_idxs]
        values = all_values[sampled_idxs]
        depths = all_depths[sampled_idxs]

        surface_points_base = depths_to_world_coords(depths, rays, camera_pos)

        empty_points, empty_points_weights = importance_sample_empty_points(
            surface_points_base, depths, camera_pos)

        if noisy:
            depth_noise = 0.07 if noisy else None
            surface_points = depths_to_world_coords(depths, rays, camera_pos, depth_noise=depth_noise)
        else:
            surface_points = surface_points_base

        if self.do_frustum_culling:
            # Cull those points which lie outside the input view
            visible = frustum_cull(surface_points, example['camera_pos'], example['input_rays'])

            print(visible.shape, surface_points.shape)

            surface_points = surface_points[visible]
            empty_points = empty_points[visible]
            values = values[visible]
            depths = depths[visible]
            rays = rays[visible]

        # 3d world coordinates of the points where the training rays encounter a surface
        example['surface_points'] = surface_points
        # 3d world coordinates of some points sampled between the camera and the surface points
        example['empty_points'] = empty_points
        # Importance weights of the empty points
        example['empty_points_weights'] = empty_points_weights
        example['values'] = values  # Color values of the surface points
        example['rays'] = rays  # Ray directions of the training rays

        if self.mode != 'train':
            mask_idx = imageio.imread(os.path.join(self.path, 'masks', f'masks_{scene_idx}_{view_idx}.png'))
            mask = np.zeros((240, 320, self.max_num_entities), dtype=np.uint8)
            np.put_along_axis(mask, np.expand_dims(mask_idx, -1), 1, axis=2)
            mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
            example['masks'] = mask  # Object masks for the input view

        return example



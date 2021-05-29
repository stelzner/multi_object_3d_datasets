import numpy as np
import torch

from colorsys import hsv_to_rgb


def get_camera_rays(c_pos, width=320, height=240, focal_length=0.035, sensor_width=0.032, noisy=False,
                    vertical=None, c_track_point=None):
    #c_pos = np.array((0., 0., 0.))
    # The camera is pointed at the origin
    if c_track_point is None:
        c_track_point = np.array((0., 0., 0.))

    if vertical is None:
        vertical = np.array((0., 0., 1.))

    c_dir = (c_track_point - c_pos)
    c_dir = c_dir / np.linalg.norm(c_dir)

    img_plane_center = c_pos + c_dir * focal_length

    # The horizontal axis of the camera sensor is horizontal (z=0) and orthogonal to the view axis
    img_plane_horizontal = np.cross(c_dir, vertical)
    #img_plane_horizontal = np.array((-c_dir[1]/c_dir[0], 1., 0.))
    img_plane_horizontal = img_plane_horizontal / np.linalg.norm(img_plane_horizontal)

    # The vertical axis is orthogonal to both the view axis and the horizontal axis
    img_plane_vertical = np.cross(c_dir, img_plane_horizontal)
    img_plane_vertical = img_plane_vertical / np.linalg.norm(img_plane_vertical)

    # Double check that everything is orthogonal
    def is_small(x, atol=1e-7):
        return abs(x) < atol

    assert(is_small(np.dot(img_plane_vertical, img_plane_horizontal)))
    assert(is_small(np.dot(img_plane_vertical, c_dir)))
    assert(is_small(np.dot(c_dir, img_plane_horizontal)))

    # Sensor height is implied by sensor width and aspect ratio
    sensor_height = (sensor_width / width) * height

    # Compute pixel boundaries
    horizontal_offsets = np.linspace(-1, 1, width+1) * sensor_width / 2
    vertical_offsets = np.linspace(-1, 1, height+1) * sensor_height / 2

    # Compute pixel centers
    horizontal_offsets = (horizontal_offsets[:-1] + horizontal_offsets[1:]) / 2
    vertical_offsets = (vertical_offsets[:-1] + vertical_offsets[1:]) / 2

    horizontal_offsets = np.repeat(np.reshape(horizontal_offsets, (1, width)), height, 0)
    vertical_offsets = np.repeat(np.reshape(vertical_offsets, (height, 1)), width, 1)

    if noisy:
        pixel_width = sensor_width / width
        pixel_height = sensor_height / height
        horizontal_offsets += (np.random.random((height, width)) - 0.5) * pixel_width
        vertical_offsets += (np.random.random((height, width)) - 0.5) * pixel_height

    horizontal_offsets = (np.reshape(horizontal_offsets, (height, width, 1)) *
                          np.reshape(img_plane_horizontal, (1, 1, 3)))
    vertical_offsets = (np.reshape(vertical_offsets, (height, width, 1)) *
                        np.reshape(img_plane_vertical, (1, 1, 3)))

    image_plane = horizontal_offsets + vertical_offsets

    image_plane = image_plane + np.reshape(img_plane_center, (1, 1, 3))
    c_pos_exp = np.reshape(c_pos, (1, 1, 3))
    rays = image_plane - c_pos_exp
    ray_norms = np.linalg.norm(rays, axis=2, keepdims=True)
    rays = rays / ray_norms
    return rays.astype(np.float32)


def depths_to_world_coords(depths, rays, camera_pos, depth_noise=None, noise_ratio=1.):
    #height, width = depths.shape
    #sensor_width = (0.032 / 320) * width
    #rays = get_camera_rays(camera_pos)
    # TODO: Put this code in a place that makes sense
    if depth_noise is not None:
        noise_indicator = (np.random.random(depths.shape) <= noise_ratio).astype(np.float32)
        depths = depths + noise_indicator * np.random.random(depths.shape) * depth_noise

    #rays = prep_fn(rays)

    surface_points = camera_pos + rays * np.expand_dims(depths, -1)
    return surface_points.astype(np.float32)


def importance_sample_empty_points(surface_points, depths, camera_pos, cutoff=0.98, p_near=0.5):
    num_points = surface_points.shape[0]
    rays = surface_points - camera_pos

    random_intercepts = np.random.random((num_points, 1)).astype(np.float32)

    near_indicator = np.random.binomial(1, p_near, size=(num_points, 1))
    range_bottom = near_indicator * cutoff
    range_top = cutoff + (near_indicator * (1. - cutoff))

    random_intercepts = range_bottom + (range_top - range_bottom) * random_intercepts

    noise_points = camera_pos + (random_intercepts * rays)
    weights = (cutoff * depths * (1 - near_indicator[..., 0]) * 2 +
               (1 - cutoff) * depths * near_indicator[..., 0] * 2)

    return noise_points.astype(np.float32), weights.astype(np.float32)


def zs_to_depths(zs, rays, camera_pos):
    view_axis = -camera_pos
    view_axis = view_axis / np.linalg.norm(view_axis, axis=-1, keepdims=True)
    factors = np.einsum('...i,i->...', rays, view_axis)
    depths = zs / factors
    return depths


def frustum_cull(points, camera_pos, rays, near_plane=None, far_plane=None):
    corners = [rays[0, 0], rays[0, -1], rays[-1, -1], rays[-1, 0]]
    rel_points = points - np.expand_dims(camera_pos, 0)
    included = np.ones(points.shape[0])

    for i in range(4):
        c1 = corners[i]
        c2 = corners[(i+1) % 4]


        normal = np.cross(c1, c2)
        normal /= np.linalg.norm(normal)

        d = (rel_points * np.expand_dims(normal, 0)).sum(-1)

        included = np.logical_and(included, d >= 0)

    return included


def get_clustering_colors(num_colors):
    colors = [(0., 0., 0.)]
    for i in range(num_colors):
        colors.append(hsv_to_rgb(i / num_colors, 0.45, 0.8))
    colors = np.array(colors)
    return colors



import numpy as np
import math


MEAN_CAMERA_POSITION = np.array((7.48113, -6.50764, 5.34367))


def rotate_around_z_axis(point, theta):
    # Rotate point(s) around the z axis
    result = np.zeros_like(point)
    result[..., 2] = point[..., 2]
    result[..., 0] = math.cos(theta) * point[..., 0] - math.sin(theta) * point[..., 1]
    result[..., 1] = math.sin(theta) * point[..., 0] + math.cos(theta) * point[..., 1]

    return result


def make_camera_positions(init_positions, thetas, max_jitter=0.5):
    results = np.zeros((init_positions.shape[0], len(thetas) + 1, 3))
    results[:, 0] = init_positions

    for i in range(len(thetas)):
        theta = thetas[i]

        cur_mean = rotate_around_z_axis(MEAN_CAMERA_POSITION, theta)

        jitter = np.random.uniform(-max_jitter, max_jitter, results[:, i+1].shape)
        print(jitter[:10])
        print(results.shape, cur_mean.shape, jitter.shape)
        cur_positions = np.expand_dims(cur_mean, 0) + jitter
        print(cur_positions.shape)
        results[:, i+1] = cur_positions

    return results


def make_light_jitter(num_scenes, max_jitter=1.):
    result = np.random.uniform(-max_jitter, max_jitter, (num_scenes, 3, 3))
    return result


if __name__ == '__main__':
    theta_degrees = [120, 240]
    thetas = [d * 2. * math.pi / 360. for d in theta_degrees]
    init_camera_pos = np.load('camera_pos_root_finding.npy')

    camera_pos = make_camera_positions(init_camera_pos, thetas)
    light_jitter = make_light_jitter(init_camera_pos.shape[0])

    metadata = np.load('metadata.npz')

    metadata = {k: v for k, v in metadata.items()}
    metadata['camera_pos'] = camera_pos
    metadata['light_jitter'] = light_jitter

    np.savez('completed_metadata.npz', **metadata)







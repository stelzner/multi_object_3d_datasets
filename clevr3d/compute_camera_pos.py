import argparse, os

import numpy as np
from numpy.linalg import norm
from scipy.optimize import root

import torch
import torch.nn as nn



def z_to_dist(z, c, pos):
    view_axis = -c
    obj_view = pos - c

    arccos_angle = np.dot(view_axis, obj_view) / (norm(view_axis) * norm(obj_view))
    return z / arccos_angle


def compute_z(c, pos):
    obj_view = pos - c.unsqueeze(1)
    view_axis = -c.unsqueeze(1)
    return (obj_view * view_axis).sum(-1) / torch.norm(view_axis, dim=-1)


MEAN_CAMERA = np.array((7.48113, -6.50764, 5.34367))

def get_camera_positions_sgd_batch(pos, zs, pres):
    mean_camera_position = torch.tensor(MEAN_CAMERA)

    cs = mean_camera_position.unsqueeze(0).repeat(pos.shape[0], 1).cuda()
    cs = nn.Parameter(cs)

    optim = torch.optim.Adam([cs], lr=0.01)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.5, verbose=True)

    for i in range(10000):
        pred_zs = compute_z(cs, pos)
        loss = (((pred_zs - zs)**2) * pres).sum(1)

        max_loss = loss.max().item()
        if max_loss < 1e-6:
            break

        mean_loss = loss.mean()

        optim.zero_grad()
        mean_loss.backward()
        optim.step()
        if i % 1000 == 0:
            scheduler.step()

    print('max loss', max_loss)
    print('mean loss', mean_loss)
    
    #print('max loss', loss.max().item())
    #print('mean loss', loss.mean().item())
    return cs.detach().cpu().numpy()


def get_camera_positions_sgd(metadata):
    batch_size = 512
    num_imgs, num_objs = metadata['color'].shape[:2]
    camera_positions = np.zeros((num_imgs, 3))
    for i in range(0, num_imgs, batch_size):
        pres_batch = torch.tensor(metadata['shape'][i:i+batch_size] > 0).cuda()
        pos_batch = torch.tensor(metadata['3d_coords'][i:i+batch_size]).cuda()
        z_batch = torch.tensor(metadata['pixel_coords'][i:i+batch_size, :, 2]).cuda()

        print('Idx', i)
        result_batch = get_camera_positions(pos_batch, z_batch, pres_batch)
        camera_positions[i:i+batch_size] = result_batch



def dist_function(c, pos, zs, get_all=False):
    c = c[:3]
    norm_c = norm(c)
    d = ((-np.expand_dims(c, 0)) * (pos - np.expand_dims(c, 0))).sum(-1) / norm_c - zs
    #print(d, idxs)
    #print(d_sel)
    return d
    #if not get_all:
        #return d[:3]
    #else:
        #return d
    #d2 = np.dot(-c, pos[1] - c) / norm_c - zs[1]
    #d3 = np.dot(-c, pos[2] - c) / norm_c - zs[2]
    #return np.array((d1, d2, d3))


def get_camera_positions_newton(metadata):
    pres = metadata['shape'] > 0
    positions = metadata['3d_coords']
    zs = metadata['pixel_coords'][:, :, 2]

    num_imgs, num_objs = metadata['color'].shape[:2]
    camera_positions = np.zeros((num_imgs, 3))
    errors = np.zeros((num_imgs,))
    print(MEAN_CAMERA.shape)

    for i in range(num_imgs):
        cur_pres = pres[i]
        #print(cur_pres, positions[i], zs[i])
        cur_positions = positions[i][cur_pres]
        cur_zs = zs[i][cur_pres]
        #print(cur_positions, cur_zs)
        init_c = np.zeros_like(cur_zs)
        init_c[:3] = MEAN_CAMERA
        result = root(dist_function, MEAN_CAMERA, args=(cur_positions, cur_zs), method='lm')  # , options={'maxfev': 1000, 'xtol': 1e-7})
        camera_positions[i] = result.x[:3]
        #print(result.success, result.message)
        errors[i] = np.abs(dist_function(camera_positions[i], cur_positions, cur_zs, get_all=True)).max()
        if errors[i] > 0.0005:
            print(i, errors[i])
            print(result.message)
        if i % 100 == 0:
            print(i)

    print('Max error', errors.max())

    return camera_positions, errors


def compute_camera_positions(metadata_file):
    metadata = np.load(metadata_file)

    camera_positions, errors = get_camera_positions_newton(metadata)
    np.save('camera_pos_root_finding.npy', camera_positions)
    np.save('errors_root_finding.npy', errors)
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('metadata_file', help='JSON file specifying scene')

    args = parser.parse_args()

    compute_camera_positions(args.metadata_file)


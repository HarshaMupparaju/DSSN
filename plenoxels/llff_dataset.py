import glob
import os
from typing import Tuple, List
from plenoxels.intrinsics import Intrinsics
from plenoxels.ray_utils import center_poses
import numpy as np
import torch
from numpy import savetxt


def saving_in_ext(datadir: 'data/dynerf/cut_roasted_beef', downsample: float = 2, near_scaling: float = 0) -> Tuple[np.ndarray, np.ndarray, Intrinsics]:
    poses_bounds = np.load(os.path.join(datadir, 'poses_bounds.npy'))  # (N_images, 17)
    poses, near_fars, intrinsics = _split_poses_bounds(poses_bounds)
# [:poses_bounds.shape[0]//data_part,]
# save_poses = np.repeat(np.array([0,0,0,1]),20,axis =1)
    b = np.swapaxes(np.dstack([np.eye(4)]*20),0,2)

    # Step 1: rescale focal length according to training resolution
    intrinsics.scale(1 / downsample)

    # Step 2: correct poses
    # Original poses has rotation in form "down right back", change to "right up back"
    # See https://github.com/bmild/nerf/issues/34
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)


# saving extransic file
    b[:, :3, :] = poses
    w2c = b
    extrensic = savetxt(os.path.join(datadir, 'extrensic.csv'), w2c.reshape(20,-1),delimiter=',')
# intrinsic file
    intrinsic_matrix = np.array([intrinsics.focal_x,0,intrinsics.center_x,0,intrinsics.focal_y,intrinsics.center_y,0,0,1]).reshape(3,3)
    intrinsic_matrix = np.swapaxes(np.swapaxes(np.repeat(intrinsic_matrix[:,:,np.newaxis],20,axis=2),0,2),1,2)
    intrinsic_file = savetxt(os.path.join(datadir, 'intrensic.csv'), intrinsic_matrix.reshape(20,-1),delimiter=',')





def _split_poses_bounds(poses_bounds: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Intrinsics]:
    poses = poses_bounds[:, :15].reshape(-1, 3, 5)  # (N_images, 3, 5)
    near_fars = poses_bounds[:, -2:]  # (N_images, 2)
    H, W, focal = poses[0, :, -1]  # original intrinsics, same for all images
    intrinsics = Intrinsics(
        width=W, height=H, focal_x=focal, focal_y=focal, center_x=W / 2, center_y=H / 2)
    return poses[:, :, :4], near_fars, intrinsics


def load_llff_poses_helper(datadir: str, downsample: float, near_scaling: float) -> Tuple[np.ndarray, np.ndarray, Intrinsics]:
    poses_bounds = np.load(os.path.join(datadir, 'poses_bounds.npy'))  # (N_images, 17)
    poses, near_fars, intrinsics = _split_poses_bounds(poses_bounds) #[:poses_bounds.shape[0]//data_part,]

    # Step 1: rescale focal length according to training resolution
    intrinsics.scale(1 / downsample)

    # Step 2: correct poses
    # Original poses has rotation in form "down right back", change to "right up back"
    # See https://github.com/bmild/nerf/issues/34
    poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1)

    # (N_images, 3, 4) exclude H, W, focal
    poses, pose_avg = center_poses(poses)   # this is in x,y,z convention and this is w2c matrix

    # Step 3: correct scale so that the nearest depth is at a little more than 1.0
    # See https://github.com/bmild/nerf/issues/34
    near_original = np.min(near_fars)
    scale_factor = near_original * near_scaling  # 0.75 is the default parameter
    # the nearest depth is at 1/0.75=1.33
    near_fars /= scale_factor
    poses[..., 3] /= scale_factor

    return poses, near_fars, intrinsics


def load_llff_poses(datadir: str,
                    downsample: float,
                    split: str,
                    hold_every: int,
                    near_scaling: float = 0.75) -> Tuple[
                        List[str], torch.Tensor, torch.Tensor, Intrinsics]:
    int_dsample = int(downsample)
    if int_dsample != downsample or int_dsample not in {4, 8}:
        raise ValueError(f"Cannot downsample LLFF dataset by {downsample}.")

    poses, near_fars, intrinsics = load_llff_poses_helper(datadir, downsample, near_scaling)

    image_paths = sorted(glob.glob(os.path.join(datadir, f'images_{int_dsample}/*')))
    assert poses.shape[0] == len(image_paths), \
        'Mismatch between number of images and number of poses! Please rerun COLMAP!'

    # Take training or test split
    i_test = np.arange(0, poses.shape[0], hold_every)
    img_list = i_test if split != 'train' else list(set(np.arange(len(poses))) - set(i_test))
    img_list = np.asarray(img_list)

    image_paths = [image_paths[i] for i in img_list]
    poses = torch.from_numpy(poses[img_list]).float()
    near_fars = torch.from_numpy(near_fars[img_list]).float()


    return image_paths, poses, near_fars, intrinsics


def load_llff_images(image_paths: List[str], intrinsics: Intrinsics, split: str):
    all_rgbs: List[torch.Tensor] = \
        parallel_load_images(
        tqdm_title=f'Loading {split} data',
        dset_type='llff',
        data_dir='/',  # paths from glob are absolute
        num_images=len(image_paths),
        paths=image_paths,
        out_h=intrinsics.height,
        out_w=intrinsics.width,
    )
    return torch.stack(all_rgbs, 0)


#saving_in_ext(datadir = '/home/kapilchoudhary/Downloads/DKnerf/data/dynerf/cut_roasted_beef',downsample=2, near_scaling=0)
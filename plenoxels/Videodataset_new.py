import numpy as np
import logging as log
import os
import glob
from numpy import ndarray
from torch import Tensor
from .datasets.loding_csv import OF_paths
from .ray_utils import (
    create_meshgrid, stack_camera_dirs, get_rays, generate_spiral_path
)

import imageio.v3 as iio
from typing import Any, List,Optional, Tuple
# import pandas as pd
import torch
import torch.utils.data
from plenoxels.intrinsics import Intrinsics
from plenoxels.llff_dataset import load_llff_poses_helper
from plenoxels.base_dataset import BaseDataset
def load_llffvideo_poses(datadir: str,
                         downsample: float,
                         split: str,
                         near_scaling: float,data_part =2) :
    """Load poses and metadata for LLFF video.

    Args:
        datadir (str): Directory containing the videos and pose information
        downsample (float): How much to downsample videos. The default for LLFF videos is 2.0
        split (str): 'train' or 'test'.
        near_scaling (float): How much to scale the near bound of poses.

    Returns:
        Tensor: A tensor of size [N, 4, 4] containing c2w poses for each camera.
        Tensor: A tensor of size [N, 2] containing near, far bounds for each camera.
        Intrinsics: The camera intrinsics. These are the same for every camera.
        List[str]: List of length N containing the path to each camera's data.
    """
    poses, near_fars, intrinsics = load_llff_poses_helper(datadir, downsample, near_scaling)
    # poses, near_fars, intrinsics = poses, near_fars, intrinsics

    videopaths = np.array(glob.glob(os.path.join(datadir, '*.mp4')))

    assert poses.shape[0] == len(videopaths), \
        'Mismatch between number of cameras and number of poses!'
    videopaths.sort()

    # The first camera is reserved for testing, following https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0
    if split == 'train':
        split_ids = np.arange(1, poses.shape[0], 7)
    elif split == 'test':
        split_ids = np.array([1])
        # split_ids = np.arange(1, poses.shape[0], 7)
    else:
        split_ids = np.arange(poses.shape[0])

    poses = torch.from_numpy(poses[split_ids])
    near_fars = torch.from_numpy(near_fars[split_ids])
    videopaths = videopaths[split_ids].tolist()

    return poses, near_fars, intrinsics, videopaths, split_ids


def _load_video_1cam(idx: int,
                     paths: List[str],
                     poses: torch.Tensor,
                     out_h: int,
                     out_w: int,
                     load_every: int = 1
                     ):  # -> Tuple[List[torch.Tensor], torch.Tensor, List[int]]:
    filters = [
        ("scale", f"w={out_w}:h={out_h}")
    ]
    all_frames = iio.imread(
        paths[idx], plugin='pyav', format='rgb24', constant_framerate=True, thread_count=2,
        filter_sequence=filters,)
    imgs, timestamps = [], []
    for frame_idx, frame in enumerate(all_frames):
        if frame_idx % load_every != 0:
            continue
        if frame_idx >= 300:  # Only look at the first 10 seconds
            break
        # Frame is np.ndarray in uint8 dtype (H, W, C)
        imgs.append(
            torch.from_numpy(frame)
        )
        timestamps.append(frame_idx)
    imgs = torch.stack(imgs, 0)
    med_img, _ = torch.median(imgs, dim=0)  # [h, w, 3]
    return \
        (imgs,
            poses[idx].expand(len(timestamps), -1, -1),
            med_img,
            torch.tensor(timestamps, dtype=torch.int32))


def parallel_load_images(tqdm_title,
                         dset_type: str,
                         num_images: int,
                         **kwargs) -> List[Any]:
    max_threads = 10

    if  dset_type == 'video':
        fn = 1
        max_threads =4
    else:
        raise ValueError(dset_type)
    outputs = []
    if fn == 1:
        for i in range(num_images):
            fn = _load_video_1cam(idx=i, **kwargs)
            if i is not None:
                outputs.append(fn)

    return outputs


def load_llffvideo_data(videopaths: List[str],
                        cam_poses: torch.Tensor,
                        intrinsics: Intrinsics,
                        split: str,
                        keyframes: bool,
                        keyframes_take_each: Optional[int] = None,
                        ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if keyframes and (keyframes_take_each is None or keyframes_take_each < 1):
        raise ValueError(f"'keyframes_take_each' must be a positive number, "
                         f"but is {keyframes_take_each}.")

    loaded =parallel_load_images(
            dset_type="video",
            tqdm_title=f"Loading {split} data",
            num_images=len(videopaths),#%2
            paths=videopaths,#[:len(videopaths)%2],
            poses=cam_poses,
            out_h=intrinsics.height,
            out_w=intrinsics.width,
            load_every=keyframes_take_each if keyframes else 1,
        )

    imgs, poses, median_imgs, timestamps = zip(*loaded)

    timestamps = torch.cat(timestamps, 0)  # [N]
    poses = torch.cat(poses, 0)            # [N, 3, 4]
    imgs = torch.cat(imgs, 0)              # [N, h, w, 3]
    median_imgs = torch.stack(median_imgs, 0)  # [num_cameras, h, w, 3]

    return poses, imgs, timestamps, median_imgs


class Video360Dataset(BaseDataset):
    len_time: int
    max_cameras: Optional[int]
    max_tsteps: Optional[int]
    timestamps: Optional[torch.Tensor]

    def __init__(self,
                 datadir: str,
                 split: str,
                 batch_size: Optional[int] = None,
                 ratio_of_optical_supervisied_train_data:Optional[float] =None,
                 downsample: float = 1.0,
                 keyframes: bool = False,
                 max_cameras: Optional[int] = None,
                 max_tsteps: Optional[int] = None,
                 isg: bool = False,
                 contraction: bool = False,
                 ndc: bool = False,
                 scene_bbox: Optional[List] = None,
                 near_scaling: float = 0.9,
                 ndc_far: float = 2.6,
                 frame_diff_for_motion = 10
                 ):
            self.sep_in_train_vid = 7
            self.keyframes = keyframes
            self.max_cameras = max_cameras
            self.max_tsteps = max_tsteps
            self.downsample = downsample
            self.isg = isg
            self.ist = False
            # self.lookup_time = False
            self.per_cam_near_fars = None
            self.global_translation = torch.tensor([0, 0, 0])
            self.global_scale = torch.tensor([1, 1, 1])
            self.near_scaling = near_scaling
            self.ndc_far = ndc_far
            self.median_imgs = None
            self.per_cam_poses = None

            if "lego" in datadir or "dnerf" in datadir:
                dset_type = "synthetic"

            else:
                dset_type = "llff"

            if dset_type == "llff":
                if split == "render":
                    assert ndc, "Unable to generate render poses without ndc: don't know near-far."
                    per_cam_poses, per_cam_near_fars, intrinsics, _, video_num = load_llffvideo_poses(
                        datadir, downsample=self.downsample, split='all', near_scaling=self.near_scaling)
                    render_poses = generate_spiral_path(
                        per_cam_poses.numpy(), per_cam_near_fars.numpy(), n_frames=300,
                        n_rots=2, zrate=0.5, dt=self.near_scaling, percentile=60)
                    self.per_cam_poses = per_cam_poses.float()
                    self.video_num = video_num
                    self.poses = torch.from_numpy(render_poses).float()
                    self.per_cam_near_fars = torch.tensor([[0.4, self.ndc_far]])
                    timestamps = torch.linspace(0, 299, len(self.poses))
                    imgs = None

                else:
                    per_cam_poses, per_cam_near_fars, intrinsics, videopaths, video_num = load_llffvideo_poses(
                        datadir, downsample=self.downsample, split=split, near_scaling=self.near_scaling)
                    if split == 'test':
                        keyframes = False
                    poses, imgs, timestamps, self.median_imgs = load_llffvideo_data(
                        videopaths=videopaths, cam_poses=per_cam_poses, intrinsics=intrinsics,
                        split=split, keyframes=keyframes, keyframes_take_each=30)

                    self.poses = poses.float()
                    self.per_cam_poses = per_cam_poses.float()
                    self.video_num = video_num

                    if contraction:
                        self.per_cam_near_fars = per_cam_near_fars.float()
                    else:
                        self.per_cam_near_fars = torch.tensor(
                            [[0.0, self.ndc_far]]).repeat(per_cam_near_fars.shape[0], 1)

                # These values are tuned for the salmon video
                self.global_translation = torch.tensor([0, 0, 2.])
                self.global_scale = torch.tensor([0.5, 0.6, 1])
                self.timestamps_max = timestamps.max()
                # Normalize timestamps between -1, 1
                timestamps = (timestamps.float() / self.timestamps_max) * 2 - 1

            self.timestamps = timestamps
            self.intrinsics = intrinsics
            self.frame_diff_for_motion = frame_diff_for_motion

            if split == 'train':
                self.timestamps = self.timestamps[:, None, None].repeat(
                        1, intrinsics.height, intrinsics.width).reshape(-1)  # [n_frames * h * w]

            assert self.timestamps.min() >= -1.0 and self.timestamps.max() <= 1.0, "timestamps out of range."

            if imgs is not None and imgs.dtype != torch.uint8:
                imgs = (imgs * 255).to(torch.uint8)

            if self.median_imgs is not None and self.median_imgs.dtype != torch.uint8:
                    self.median_imgs = (self.median_imgs * 255).to(torch.uint8)

            if split == 'train':
                imgs = imgs.view(-1, imgs.shape[-1])

            elif imgs is not None:
                imgs = imgs.view(-1, intrinsics.height * intrinsics.width, imgs.shape[-1])

                # ISG/IST weights are computed on 4x subsampled data.
            weights_subsampled = int(4 / downsample)
            if scene_bbox is not None:
                    scene_bbox = torch.tensor(scene_bbox)

            # else:
            #         scene_bbox = get_bbox(datadir, is_contracted=contraction, dset_type=dset_type)
            super().__init__(
                datadir=datadir,
                split=split,
                batch_size=batch_size,
                is_ndc=ndc,
                is_contracted=contraction,
                scene_bbox=scene_bbox,
                rays_o=None,
                rays_d=None,
                intrinsics=intrinsics,
                imgs=imgs,
                sampling_weights=None,  # Start without importance sampling, by default
                weights_subsampled=weights_subsampled,
                ratio_of_optical_supervisied_train_data=ratio_of_optical_supervisied_train_data
            )
            if split == "train" and dset_type == 'llff':
                self.csv_data, self.f1, self.c1, self.f2, self.c2 = OF_paths()

                data = [self.csv_data[i] for i in range(len(self.csv_data))]

                x1 = [torch.tensor(data[i]["x1"].values) for i in
                      range(len(data))]  # TODO : make the csv files read without header
                no_of_samples_per_csv = [len(x1[i]) for i in range(len(x1))]
                x1 = torch.cat(x1)
                y1 = torch.cat([torch.tensor(data[i]["y1"].values) for i in range(len(data))])
                x2 = torch.cat([torch.tensor(data[i]["x2"].values) for i in range(len(data))])
                y2 = torch.cat([torch.tensor(data[i]["y2"].values) for i in range(len(data))])
                # x1 = x1[:int(csv_rand_points/2)]

                f1 = torch.tensor([self.f1[i] for i in range(len(self.f1))])  #

                f1_time = (torch.cat([f1[i].repeat(no_of_samples_per_csv[i]) for i in range(len(f1))]) / self.timestamps_max) * 2 - 1
 
                c1_id = torch.tensor([self.c1[i] // self.sep_in_train_vid for i in range(len(self.c1))])
                c1_id_rep = torch.cat([c1_id[i].repeat(no_of_samples_per_csv[i]) for i in range(len(c1_id))])

                f2 = torch.tensor([self.f2[i] for i in range(len(self.f2))])
                f2_time = (torch.cat([f2[i].repeat(no_of_samples_per_csv[i]) for i in range(len(f2))]) / self.timestamps_max) * 2 - 1

                c2_id = torch.tensor([self.c2[i] // self.sep_in_train_vid for i in range(len(self.c2))])
                c2_id_rep = torch.cat([c2_id[i].repeat(no_of_samples_per_csv[i]) for i in range(len(c2_id))])
                # csv_rand_points = self.batch_size - len(index)
                # c = torch.cat([c2_id_rep[:int(csv_rand_points / 2)], c1_id_rep[:int(csv_rand_points / 2)]])

                # iid = 300*c1_id+f1
                # iid1 = 300*c2_id+f2
                image_id1 = 300 * c1_id_rep + (f1_time + 1) * self.timestamps_max / 2
                image_id2 = 300 * c2_id_rep + (f2_time + 1) * self.timestamps_max / 2
                c2w1 = self.poses[image_id1.long()]
                c2w2 = self.poses[image_id2.long()]
                
                c2w1_homo = torch.cat((c2w1,torch.tensor([0,0,0,1]).reshape(1,4).repeat(c2w1.shape[0],1,1)),1)
                w2c1_homo = torch.linalg.inv(c2w1_homo)

                c2w2_homo = torch.cat((c2w2,torch.tensor([0,0,0,1]).reshape(1,4).repeat(c2w2.shape[0],1,1)),1)
                w2c2_homo = torch.linalg.inv(c2w2_homo)

                #v is source or intial camera, u is destination or final camera
                
                x_u = torch.cat((x1,x2))
                y_u = torch.cat((y1,y2))
                f_u = torch.cat((f1_time,f2_time))
                c_u = torch.concat((c1_id_rep,c2_id_rep))
                c2w_u_homo = torch.cat((c2w1_homo,c2w2_homo))
                w2c_u_homo = torch.cat((w2c2_homo,w2c1_homo))

                x_v = torch.cat((x2,x1))
                y_v = torch.cat((y2,y1))
                f_v = torch.cat((f2_time,f1_time))
                c_v = torch.cat((c2_id_rep,c1_id_rep))
                c2w_v_homo = torch.cat((c2w2_homo,c2w1_homo))
                w2c_v_homo = torch.cat((w2c1_homo,w2c2_homo))

                self.points_cameras_concat = {
                                                'x_u': x_u,
                                                'y_u': y_u,
                                                'x_v': x_v,
                                                'y_v': y_v,
                                                'f_u': f_u, 
                                                'f_v': f_v, 
                                                'c_u': c_u, 
                                                'c_v': c_v, 
                                                'c2w_u': c2w_u_homo, 
                                                'w2c_u': w2c_u_homo, 
                                                'c2w_v': c2w_v_homo,
                                                'w2c_v': w2c_v_homo
                                                }
                # adding csv and rand data
            self.isg_weights = None
            self.ist_weights = None
            # if split == "train" and dset_type == 'llff':  # Only use importance sampling with DyNeRF videos
            #     if os.path.exists(os.path.join(datadir, f"isg_weights.pt")):
            #         self.isg_weights = torch.load(os.path.join(datadir, f"isg_weights.pt"))
            #         log.info(f"Reloaded {self.isg_weights.shape[0]} ISG weights from file.")
            #
            # if self.isg:
            #     self.enable_isg()

            log.info(f"VideoDataset contracted={self.is_contracted}, ndc={self.is_ndc}. "
                     f"Loaded {self.split} set from {self.datadir}: "
                     f"{len(self.poses)} images of size {self.img_h}x{self.img_w}. "
                     f"Images loaded: {self.imgs is not None}. "
                     f"{len(torch.unique(timestamps))} timestamps. Near-far: {self.per_cam_near_fars}. "
                     f"ISG={self.isg}, IST={self.ist}, weights_subsampled={self.weights_subsampled}. "
                     f"Sampling without replacement={self.use_permutation}. {intrinsics}")

    def enable_isg(self):
        self.isg = True
        self.ist = False
        self.sampling_weights = self.isg_weights
        log.info(f"Enabled ISG weights.")

    def switch_isg2ist(self):
        self.isg = False
        self.ist = True
        self.sampling_weights = self.ist_weights
        log.info(f"Switched from ISG to IST weights.")

    def __getitem__(self, index):
        h = self.intrinsics.height
        w = self.intrinsics.width
        dev = "cpu"
        if self.split == 'train':
            index = self.get_rand_ids(index)  # [batch_size // (weights_subsampled**2)] # changed on 7 sep
            # a = int(self.batch_size * self.ratio_of_optical_supervisied_train_data)
            # csv_rand_points = a if a + len(index) == self.batch_size else a + 1
            index_csv = self.get_csv_rand_ids()  # changed on 7 sep
            if self.weights_subsampled == 1 or self.sampling_weights is None:
                # Nothing special to do, either weights_subsampled = 1, or not using weights.
                image_id = torch.div(index, h * w, rounding_mode='floor')
                y = torch.remainder(index, h * w).div(w, rounding_mode='floor')
                x = torch.remainder(index, h * w).remainder(w)
            x, y = x + 0.5, y + 0.5
            #csv data loading

            data = [self.csv_data[i] for i in index_csv]

            x1 = [torch.tensor(data[i]["x1"].values) for i in range(len(data))] #TODO : make the csv files read without header
            no_of_samples_per_csv = [len(x1[i]) for i in range(len(x1))]
            x1 = torch.cat(x1)
            y1 = torch.cat([torch.tensor(data[i]["y1"].values) for i in range(len(data))])
            x2 = torch.cat([torch.tensor(data[i]["x2"].values) for i in range(len(data))])
            y2 = torch.cat([torch.tensor(data[i]["y2"].values) for i in range(len(data))])
            # x1 = x1[:int(csv_rand_points/2)]

            f1 = torch.tensor([self.f1[i] for i in index_csv])  #
            f1_time = (torch.cat([f1[i].repeat(no_of_samples_per_csv[i]) for i in range(len(f1))]) / self.timestamps_max) * 2 - 1

            c1_id = torch.tensor([self.c1[i] // self.sep_in_train_vid for i in index_csv])
            c1_id_rep = torch.cat([c1_id[i].repeat(no_of_samples_per_csv[i]) for i in range(len(c1_id))])

            f2 = torch.tensor([self.f2[i] for i in index_csv])
            f2_time = (torch.cat([f2[i].repeat(no_of_samples_per_csv[i]) for i in range(len(f1))]) / self.timestamps_max) * 2 - 1

            c2_id = torch.tensor([self.c2[i] // self.sep_in_train_vid for i in index_csv])
            c2_id_rep = torch.cat([c2_id[i].repeat(no_of_samples_per_csv[i]) for i in range(len(c2_id))])
            csv_rand_points = self.batch_size-len(index)
            c = torch.cat([c2_id_rep[:int(csv_rand_points / 2)], c1_id_rep[:int(csv_rand_points / 2)]])

            # iid = 300*c1_id+f1
            # iid1 = 300*c2_id+f2
            image_id1 = 300 * c1_id_rep + (f1_time+1)*self.timestamps_max/2
            image_id2 = 300 * c2_id_rep + (f2_time+1)*self.timestamps_max/2

            # adding csv and rand data
            x1_1 = torch.cat([x1[:int(csv_rand_points / 2)], x2[:int(csv_rand_points / 2)]])
            x = torch.cat([x, x1_1])
            # x = x1_
            y1_2 = torch.cat([y1[:int(csv_rand_points / 2)], y2[:int(csv_rand_points / 2)]])
            y = torch.cat([y, y1_2])
            # y = y1_
            im = self.imgs.reshape(-1, h, w, 3)
            im_idx1 = torch.cat([image_id1[:int(csv_rand_points / 2)], image_id2[:int(csv_rand_points / 2)]])
            # im_idx = im_idx1*h*w + w*x1_ + y1_
            # im_idx = im_idx.long()
            csv_rand_points = len(c)

        else:
            image_id = [index]
            x, y = create_meshgrid(height=h, width=w, dev=dev, add_half=True, flat=True)

        out = {
            "timestamps": self.timestamps[index],  # (num_rays or 1, )  #new- timestamps_of_csv_index
            "imgs": None,
        }

        if self.split == 'train':
            out["timestamps"] = torch.cat(
                [out["timestamps"], f1_time[:int(csv_rand_points // 2)], f2_time[:int(csv_rand_points // 2)]])
            num_frames_per_camera = len(self.imgs) // (len(self.per_cam_near_fars) * h * w)
            # concat with csv cam id
            camera_id = torch.div(image_id, num_frames_per_camera, rounding_mode='floor')  # (num_rays)

            image_id = torch.cat([image_id, im_idx1]).long()
            camera_id = torch.cat([camera_id, c1_id_rep[:int(csv_rand_points/2)], c2_id_rep[:int(csv_rand_points/2)]])
            # camera_id = torch.cat([c1_id_rep[:int(csv_rand_points/2)], c2_id_rep[:int(csv_rand_points/2)]])

            out['near_fars'] = self.per_cam_near_fars[camera_id, :]
        else:
            out['near_fars'] = self.per_cam_near_fars  # Only one test camera

        if self.imgs is not None:
            out['imgs'] = (self.imgs[index] / 255.0).view(-1, self.imgs.shape[-1])

        #  out['video_num'] = self.video_num
        # here the w2c is w2c for getting the ray direction and translation of the corresponsing cameras

        c2w = self.poses[image_id]  # [num_rays or 1, 3, 4]     ""  THIS IS C2W ""
        ones = torch.tensor([0, 0, 0, 1])[None, ...]
        # ones = ones
        c2w = torch.cat((c2w, ones.repeat(c2w.shape[0], 1, 1)), dim=1)

        camera_dirs = stack_camera_dirs(x, y, self.intrinsics, True)  # [num_rays, 3]
        # out['rays_o'], out['rays_d'] = get_rays(
        #     camera_dirs, c2w, ndc=self.is_ndc, ndc_near=1.0, intrinsics=self.intrinsics,normalize_rd=True)

        out['rays_o'], out['rays_d'] = get_rays(
            camera_dirs, c2w, ndc=False, ndc_near=1.0, intrinsics=self.intrinsics,
            normalize_rd=False)

             # [num_rays, 3]
        if self.split == 'train':
            out["csv_rand_points"] = csv_rand_points
            out["cam_indices_forcsv_pts"] = c
            out['imgs'] = torch.cat([out["imgs"], im[im_idx1.long(), y1_2.long(), x1_1.long()] / 255.0])
            # out['imgs'] = torch.cat([im[im_idx1.long(), y1_.long(), x1_.long()]/255.0])
        else:
            out["csv_rand_points"] = None
            out["cam_indices_forcsv_pts"] = None


        # n_c2w = self.per_cam_poses
        # ones = torch.tensor([0, 0, 0, 1])
        # ones = ones[None, ...]
        # c2w = torch.cat((n_c2w, ones.repeat(n_c2w.shape[0], 1, 1)), dim=1)  #check
        #
        # w2c = torch.linalg.inv(c2w)
        """
        the concatination is reverse why ? because we need the w2c of the camera u where it has moved from initial camera v
        and in arranging x we have x1,x2 with image index image_id1 on top half of the taken csv points and image_id corresponds to 
        where it will move , so for projection of x1 on x2 we will need the camera extensic corresponding to x2
        """
        if self.split == "train":
            im_idx2 = torch.cat([image_id2[:int(csv_rand_points / 2)], image_id1[:int(csv_rand_points / 2)]])
            c2w1 = self.poses[im_idx2.long()]  # camera pose u (the target view)

            # ones = torch.tensor([0, 0, 0, 1])
            # ones = ones[None, ...]
            # c2w = torch.cat((c2w, ones.repeat(c2w.shape[0], 1, 1)), dim=1)

            # ones = torch.tensor([0, 0, 0, 1])
            # ones = ones[None, ...]
            c2w2 = torch.cat((c2w1, ones.repeat(c2w1.shape[0], 1, 1)), dim=1)  #check

            w2c = torch.linalg.inv(c2w2)
            out['w2c'] = w2c
            out["u"] = c2w2
        #
        else:  # test
            ones = torch.tensor([0, 0, 0, 1])
            ones = ones[None, ...]
            c2w = torch.cat((c2w, ones.repeat(c2w.shape[0], 1, 1)), dim=1)  # check
            w2c = torch.linalg.inv(c2w)
            out["w2c"] = w2c
        out['c2w'] = c2w
        out["intrinsics"] = self.intrinsics
        out["frame_diff_for_motion"] = self.frame_diff_for_motion
        out['x'] = x
        out['y'] = y

        imgs = out['imgs']
        # Decide BG color
        bg_color = torch.ones((1, 3), dtype=torch.float32, device=dev)
        if self.split == 'train' and imgs.shape[-1] == 4:
            bg_color = torch.rand((1, 3), dtype=torch.float32, device=dev)
        out['bg_color'] = bg_color
        # Alpha compositing
        if imgs is not None and imgs.shape[-1] == 4:
            imgs = imgs[:, :3] * imgs[:, 3:] + bg_color * (1.0 - imgs[:, 3:])
        out['imgs'] = imgs

        return out


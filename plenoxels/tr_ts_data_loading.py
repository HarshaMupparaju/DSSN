import numpy as np
import logging as log
from typing import Dict, Any
import random
import torch
import torch.utils.data
from plenoxels.Videodataset_new import Video360Dataset

# def load_data(model_type: str, data_downsample, data_dirs, validate_only: bool, render_only: bool, **kwargs):
#     data_downsample = float(data_downsample) if data_downsample is not None else 1
#
#     if model_type == "video":
#         return tr_ts_load_data(
#             data_downsample, data_dirs, validate_only=validate_only,
#             render_only=render_only, **kwargs)

def tr_ts_load_data(data_downsample, data_dirs, validate_only, render_only, **kwargs):
    assert len(data_dirs) == 1
    od: Dict[str, Any] = {}
    if not validate_only and not render_only:
        od.update(init_tr_data(data_downsample, data_dirs[0], **kwargs))
    else:
        od.update(tr_loader=None, tr_dset=None)
    test_split = 'render' if render_only else 'test'
    od.update(init_ts_data(data_dirs[0], split=test_split, **kwargs))
    return od

def init_dloader_random(_):
    seed = torch.initial_seed() % 2**32  # worker-specific seed initialized by pytorch
    # seed = 0  # TODO SNB: remove this seed
    np.random.seed(seed)
    random.seed(seed)


def init_ts_data(data_dir, split, **kwargs):
    downsample = 2.0
    ts_dset = Video360Dataset(
        data_dir, split=split, downsample=downsample,
        max_cameras=kwargs.get('max_test_cameras', None), max_tsteps=kwargs.get('max_test_tsteps', None),
        contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox']
    )
    return {"ts_dset": ts_dset}


def init_tr_data(data_downsample, data_dir, **kwargs):
    isg = kwargs.get('isg', False)
    ist = kwargs.get('ist', False)
    keyframes = kwargs.get('keyframes', False)
    batch_size = kwargs['batch_size']
    log.info(f"Loading Video360Dataset with downsample={data_downsample}")
    tr_dset = Video360Dataset(
        data_dir, split='train', downsample=data_downsample,
        batch_size=batch_size,
        max_cameras=kwargs.get('max_train_cameras', None),
        max_tsteps=kwargs['max_train_tsteps'] if keyframes else None,
        isg=isg, keyframes=keyframes, contraction=kwargs['contract'], ndc=kwargs['ndc'],
        near_scaling=float(kwargs.get('near_scaling', 0)), ndc_far=float(kwargs.get('ndc_far', 0)),
        scene_bbox=kwargs['scene_bbox'],
        ratio_of_optical_supervisied_train_data=kwargs["ratio_of_optical_supervisied_train_data"],
        frame_diff_for_motion=kwargs.get("frame_diff_for_motion", 10)
    )

    if ist:
        tr_dset.switch_isg2ist()  # this should only happen in case we're reloading

    g = torch.Generator()
    g.manual_seed(0)

    tr_loader = torch.utils.data.DataLoader(
        tr_dset, batch_size=None,  pin_memory=True,
         generator=g)

    return {"tr_loader": tr_loader, "tr_dset": tr_dset}
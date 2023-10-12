import os
import pandas as pd
import torch
from pathlib import Path
#import pandas


def OF_paths():
    path = 'data/dynerf/Sparse_flow'
    files = Path(os.path.join(path, 'sparse_flow_csvs')).glob('*.csv')
    f1, f2, c1, c2 = [], [], [], []
    # c1_c2 = []

    dfs = list()
    for i, f in enumerate(files):
        data = pd.read_csv(f)
        stem = (f'{f.stem}')
        f1_, c1_, f2_, c2_ = int(stem[:4]), int(stem[5:7]), int(stem[8:12]), int(stem[13:])
        f1.append(f1_)
        f2.append(f2_)
        c1.append(c1_)
        c2.append(c2_)
        # c1_c2.append(c1, c2)
        dfs.append(data)

    return dfs, f1, c1, f2, c2


def projection_without_near_shift(pts, intrinsics, w2c, near=2):
    """
    Args:
        pts: (n,3,1)
        intrinsics:(instance which has all intrensic data)
        w2c: (n,4,4)

    Returns: normalised pixel cordinates

    """
    w2ndc = torch.tensor([intrinsics.focal_x/intrinsics.center_x, 0, 0, 0, intrinsics.focal_y/intrinsics.center_y, 0, 0, 0, 1]).type(torch.float32).reshape(3,3).to(pts.device)[None, ...].repeat(pts.shape[0],1,1)
    p = torch.tensor([1, 0, 0, 0, -1, 0, 0, 0, -1]).type(torch.float32).reshape(3, 3)[None, :, :].repeat(pts.shape[0],1, 1).to(pts.device)
    intrinsic_mat = torch.tensor(
        [intrinsics.focal_x, 0, intrinsics.center_x, 0, intrinsics.focal_y, intrinsics.center_y, 0, 0, 1]).type(torch.float32).reshape(3, 3).to(pts.device)[None, ...].repeat(pts.shape[0], 1, 1)
    bz = torch.tensor([0, 0, near]).type(torch.float32).reshape(3, 1).to(pts.device)[None, :, :].repeat(pts.shape[0], 1, 1)
    p_pts = p.bmm(intrinsic_mat).bmm(w2c[:, :3, :3]).bmm(torch.linalg.inv(w2ndc)).bmm(pts-bz)
    # a = torch.matmul(torch.linalg.inv(w2ndc), pts)
    # a1 = torch.matmul(w2c[:, :3, :3],a)
    # b = torch.matmul(intrinsic_mat, a1)
    # c = torch.matmul(p, b)
    norrmalize_p_pts = p_pts[:, 0:2, 0] / (p_pts[:, 2:3, 0]+1e-5)
    return norrmalize_p_pts


def projection(pts, w2c, intrinsics, near):
    '''

    Args:
        pts: n,3,1
        w2c: n,4,4
        intrinsics: object which include focal length  width height
        near: set it to 1 for ndc (near plane)

    Returns: projected points in pixel domain range(width,height)

    '''

    # p = torch.tensor([1, 0, 0, 0, -1, 0, 0, 0, -1]).type(torch.float32).reshape(3, 3)[None, :, :].repeat(pts.shape[0],1, 1).to(pts.device)
    # TODO: ADD -ve sign in [0,0] in intrinsics and the results will be correct
    intrinsic_mat = torch.tensor(
        [-intrinsics.focal_x, 0, intrinsics.center_x, 0, intrinsics.focal_y, intrinsics.center_y, 0, 0, 1]).type(
        torch.float32).reshape(3, 3).to(pts.device)[None, ...].repeat(pts.shape[0], 1, 1)

    w2ndc = torch.Tensor([intrinsics.focal_x / intrinsics.center_x, 0, 0, 0, 0, intrinsics.focal_y / intrinsics.center_y, 0, 0, 0, 0, -1, -2*near, 0, 0, -1, 0]).reshape(4, 4).repeat(pts.shape[0], 1, 1).type(torch.float32).to(pts.device)
    pts1 = torch.column_stack(
        [pts, torch.ones(pts.shape[0]).reshape(pts.shape[0], 1)[..., None].to(torch.float32).to(pts.device)])

    a = torch.linalg.inv(w2ndc).bmm(pts1)
    a1 = a[:, :, 0] / (a[:, 3:4, 0] + 1e-10)  # (n,4)

    a2 = w2c.bmm(a1[..., None])  # (n,4,1)
    p_pts = intrinsic_mat.bmm(a2[:, :3, ...])  # in a2[:,3,0] all the values will be 1
    # p_pts = p.bmm(intrinsic_mat).bmm(w2c[:, :3, :3]).bmm(a1[..., None])
    norrmalize_p_pts = p_pts[:, 0:2, 0] / (p_pts[:, 2:3, 0] + 1e-10)
    return norrmalize_p_pts

def projection_3d_poits(pts, w2c, intrinsics, near):
    intrinsic_mat = torch.tensor(
        [intrinsics.focal_x, 0, intrinsics.center_x, 0, intrinsics.focal_y, intrinsics.center_y, 0, 0, 1]).type(
        torch.float32).reshape(3, 3).to(pts.device)[None, ...].repeat(pts.shape[0], 1, 1)

    # w2ndc = torch.Tensor(
    #     [-intrinsics.focal_x / intrinsics.center_x, 0, 0, 0, -intrinsics.focal_y / intrinsics.center_y,  0, 0, 0, 1,
    #      ]).reshape(3, 3).repeat(pts.shape[0], 1, 1).type(torch.float32).to(pts.device)
    # pts1 = torch.column_stack(
    #     [pts, torch.ones(pts.shape[0]).reshape(pts.shape[0], 1)[..., None].to(torch.float32).to(pts.device)])
    a = calc_o(pts, near, intrinsics.width,intrinsics.height, intrinsics.focal_x).to(pts.device) # n,3,1
    a1 = torch.column_stack(
        [a, torch.ones(a.shape[0]).reshape(a.shape[0], 1)[..., None].to(torch.float32).to(pts.device)])
    # (n,4) TODO: there is some change

    a2 = w2c.bmm(a1)  # (n,4,1)
    p_pts = intrinsic_mat.bmm(a2[:, :3, ...])
    # p_pts = p.bmm(intrinsic_mat).bmm(w2c[:, :3, :3]).bmm(a1[..., None])
    norrmalize_p_pts = p_pts[:, 0:2, 0] / (p_pts[:, 2:3, 0] + 1e-10)
    return norrmalize_p_pts


def calc_o(o_dash,near,W,H,f_cam):
    A = torch.zeros((3,3))
    A[0,0] = -f_cam/(W/2)
    A[1,1] = -f_cam/(H/2)
    A[2,2] = 1
    A=A.reshape((1,3,3))
    A_inverse = torch.linalg.inv(A).to(o_dash.device)

    # o_dash = torch.rand((n,3,1))
    c = torch.zeros((3,1))
    c[2] = 2*near

    c = torch.reshape(c,(1,3,1)).to(o_dash.device)
    res = torch.matmul(A_inverse,(o_dash - c))
    return res

#

#read_optical_motion()
#  --log-dir logs/realdynamic/cutbeef_explicit/adding_3d_motion_consistancy/correct_l1_time_loss/3d_motion_loss_with_lr=.001 --validate-only
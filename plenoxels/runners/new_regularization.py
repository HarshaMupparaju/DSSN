import abc
import os
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim.lr_scheduler
from torch import nn

from plenoxels.models.lowrank_model import LowrankModel
from plenoxels.ops.losses.histogram_loss import interlevel_loss
from plenoxels.raymarching.ray_samplers import RaySamples

def compute_plane_tv(t, only_w: bool = False):
    batch_size, c, h, w = t.shape
    # count_h = batch_size * c * (h - 1) * w
    # count_w = batch_size * c * h * (w - 1)
    h_tv = torch.square(t[..., 1:, :] - t[..., :h - 1, :]).mean()

    w_tv = torch.square(t[..., :, 1:] - t[..., :, :w - 1]).mean()
    if only_w:
        return w_tv

    return h_tv + w_tv  # This is summing over batch and c instead of avg


def compute_plane_smoothness(t):
    batch_size, c, h, w = t.shape
    # Convolve with a second derivative filter, in the time dimension which is dimension 2
    first_difference = t[..., 1:, :] - t[..., :h - 1, :]  # [batch, c, h-1, w]
    second_difference = first_difference[..., 1:, :] - first_difference[..., :h - 2, :]  # [batch, c, h-2, w]
    # Take the L2 norm of the result
    return torch.square(second_difference).mean()


class Regularizer():
    def __init__(self, reg_type, initialization):
        self.reg_type = reg_type
        self.initialization = initialization
        self.weight = float(self.initialization)
        self.last_reg = None

    def step(self, global_step):
        pass

    def report(self, d):
        if self.last_reg is not None:
            d[self.reg_type].update(self.last_reg.item())

    def regularize(self, *args, **kwargs) -> torch.Tensor:
        out = self._regularize(*args, **kwargs) * self.weight
        self.last_reg = out.detach()
        return out

    @abc.abstractmethod
    def _regularize(self, *args, **kwargs) -> torch.Tensor:
        raise NotImplementedError()

    def __str__(self):
        return f"Regularizer({self.reg_type}, weight={self.weight})"


class PlaneTV(Regularizer):
    def __init__(self, initial_value, what: str = 'field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        name = f'planeTV-{what[:2]}'
        super().__init__(name, initial_value)
        self.what = what

    def step(self, global_step):
        pass

    def _regularize(self, model: LowrankModel, **kwargs):
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field' and model.disp_grid_used is not None:
            multi_res_grids = model.field.grids
            disp_multi_res_grids = model.disp_field.disp_grids


        elif self.what == 'field' and model.disp_grid_used is None:
            multi_res_grids = model.field.grids
            # disp_multi_res_grids = model.disp_field.grids

        elif self.what == 'proposal_network':
            multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)
        total = 0
        # Note: input to compute_plane_tv should be of shape [batch_size, c, h, w]
        num_planes = 0
        for grids in multi_res_grids:
            if len(grids) == 3:
                spatial_grids = [0, 1, 2]
            else:
                spatial_grids = [0, 1, 3]  # These are the spatial grids; the others are spatiotemporal

            for grid_id, grid in enumerate(grids):
                if grid_id in spatial_grids:
                    total += compute_plane_tv(grid)
                else:
                    # Space is the last dimension for space-time planes.
                    total += compute_plane_tv(grid, only_w=True)
                num_planes += 1

            #  '''          for grid_id in spatial_grids:
            #     total += compute_plane_tv(grids[grid_id])
            # for grid in grids:
            #     # grid: [1, c, h, w]
            #     total += compute_plane_tv(grid)'''

        # extra for the displacement grids
        if self.what == 'field' and model.disp_grid_used is not None:
            for grids in disp_multi_res_grids:
                if len(grids) == 3:
                    spatial_grids = [0, 1, 2]
                else:
                    spatial_grids = [0, 1, 3]  # These are the spatial grids; the others are spatiotemporal
                for grid_id, grid in enumerate(grids):
                    if grid_id in spatial_grids:
                        total += compute_plane_tv(grid)
                    else:
                        # Space is the last dimension for space-time planes.
                        total += compute_plane_tv(grid, only_w=True)
                    num_planes += 1
        if not torch.isfinite(torch.as_tensor(total)) and num_planes == 0:
            print('CP08: Nan Found')
            return torch.tensor(0)
        else:
            return total/num_planes



class TimeSmoothness(Regularizer):
    def __init__(self, initial_value, what: str = 'field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        name = f'time-smooth-{what[:2]}'
        super().__init__(name, initial_value)
        self.what = what

    def _regularize(self, model: LowrankModel, **kwargs) -> torch.Tensor:
        multi_res_grids: Sequence[nn.ParameterList]
        if self.what == 'field' and model.disp_grid_used is not None:
            multi_res_grids = model.field.grids
            disp_multi_res_grids = model.disp_field.disp_grids

        elif self.what == 'field' and model.disp_grid_used is None:
            multi_res_grids = model.field.grids

        elif self.what == 'proposal_network':
            multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)
        total = 0
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        for grids in multi_res_grids:
            if len(grids) == 3:
                time_grids = []
            else:
                time_grids = [2, 4, 5]
            for grid_id in time_grids:
                total += compute_plane_smoothness(grids[grid_id])
        # #for the displacement field grids
        if self.what == 'field' and model.disp_grid_used is not None:
            for grids in disp_multi_res_grids:
                if len(grids) == 3:
                    time_grids = []
                else:
                    time_grids = [2, 4, 5]
                for grid_id in time_grids:
                    total += compute_plane_smoothness(grids[grid_id])
        if not torch.isfinite(torch.as_tensor(total)):
            print('CP07: Nan Found')
            return torch.tensor(0)
        else:
            return torch.as_tensor(total)


class HistogramLoss(Regularizer):
    def __init__(self, initial_value):
        super().__init__('histogram-loss', initial_value)

        self.visualize = False
        self.count = 0

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        if self.visualize:
            if self.count % 500 == 0:
                prop_idx = 0
                fine_idx = 1
                # proposal info
                weights_proposal = model_out["weights_list"][prop_idx].detach().cpu().numpy()
                spacing_starts_proposal = model_out["ray_samples_list"][prop_idx].spacing_starts
                spacing_ends_proposal = model_out["ray_samples_list"][prop_idx].spacing_ends
                sdist_proposal = torch.cat([
                    spacing_starts_proposal[..., 0],
                    spacing_ends_proposal[..., -1:, 0]
                ], dim=-1).detach().cpu().numpy()

                # fine info
                weights_fine = model_out["weights_list"][fine_idx].detach().cpu().numpy()
                spacing_starts_fine = model_out["ray_samples_list"][fine_idx].spacing_starts
                spacing_ends_fine = model_out["ray_samples_list"][fine_idx].spacing_ends
                sdist_fine = torch.cat([
                    spacing_starts_fine[..., 0],
                    spacing_ends_fine[..., -1:, 0]
                ], dim=-1).detach().cpu().numpy()

                for i in range(10):  # plot 10 rays
                    fix, ax1 = plt.subplots()

                    delta = np.diff(sdist_proposal[i], axis=-1)
                    ax1.bar(sdist_proposal[i, :-1], weights_proposal[i].squeeze() / delta, width=delta, align="edge",
                            label='proposal', alpha=0.7, color="b")
                    ax1.legend()
                    ax2 = ax1.twinx()

                    delta = np.diff(sdist_fine[i], axis=-1)
                    ax2.bar(sdist_fine[i, :-1], weights_fine[i].squeeze() / delta, width=delta, align="edge",
                            label='fine', alpha=0.3, color='r')
                    ax2.legend()
                    os.makedirs(f'histogram_loss/{self.count}', exist_ok=True)
                    plt.savefig(f'./histogram_loss/{self.count}/batch_{i}.png')
                    plt.close()
                    plt.cla()
                    plt.clf()
            self.count += 1
        loss = interlevel_loss(model_out['weights_list'], model_out['ray_samples_list'])
        if not torch.isfinite(loss):
            print('CP06: Nan Found')
            return torch.tensor([0])
        else:
            return loss


class L1ProposalNetwork(Regularizer):
    def __init__(self, initial_value):
        super().__init__('l1-proposal-network', initial_value)

    def _regularize(self, model: LowrankModel, **kwargs) -> torch.Tensor:
        grids = [p.grids for p in model.proposal_networks]
        total = 0.0
        for pn_grids in grids:
            for grid in pn_grids:
                total += torch.abs(grid).mean()
        return torch.as_tensor(total)


class L2motionloss(Regularizer):
    def __init__(self, initial_value, what: str = 'r_field'):
        name = f'l2-3d motion loss -{what[:3]}'
        super().__init__(name, initial_value)
        self.what = what

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        loss = torch.linalg.norm(model_out["rev_disp"]-model_out['pts'], dim=-1).mean()
        # if torch.isnan(loss).sum() != 0 or torch.isinf(loss).sum()!=0:
        if not torch.isfinite(loss):
            print('CP05: Nan Found')
            return torch.tensor(0)

        else:
            return loss

# 3d motion loss according to volume rendered weighted 3d points from grid and original 'pts'


class L1volume_rendered_point_loss(): # aug24
    def __init__(self, initial_value, what: str = 'r_field'):
        name = f'l1-3d motion loss -{what[:3]}'
        super().__init__(name, initial_value)
        self.what = what

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        return torch.linalg.norm(model_out["rev_disp"].reshape(-1,3)-model_out['pts'].reshape(-1,3), dim=1).mean()


class DepthTV(Regularizer):
    def __init__(self, initial_value):
        super().__init__('tv-depth', initial_value)

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        depth = model_out['depth']
        tv = compute_plane_tv(
            depth.reshape(64, 64)[None, None, :, :]
        )
        return tv


class L1TimePlanes(Regularizer):
    def __init__(self, initial_value, what='field'):
        if what not in {'field', 'proposal_network'}:
            raise ValueError(f'what must be one of "field" or "proposal_network" '
                             f'but {what} was passed.')
        super().__init__(f'l1-time-{what[:2]}', initial_value)
        self.what = what

    def _regularize(self, model: LowrankModel, **kwargs) -> torch.Tensor:
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        if self.what == 'field' and model.disp_grid_used is not None:
            multi_res_grids = model.field.grids
            disp_multi_res_grids = model.disp_field.disp_grids

        elif self.what == 'field' and model.disp_grid_used is None:
            multi_res_grids = model.field.grids

        elif self.what == 'proposal_network':
            multi_res_grids = [p.grids for p in model.proposal_networks]
        else:
            raise NotImplementedError(self.what)

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        # for disp network:
        if self.what == 'field' and model.disp_grid_used is not None:
            for grids in disp_multi_res_grids:
                if len(grids) == 3:
                    continue
                else:
                    # These are the spatiotemporal grids
                    spatiotemporal_grids = [2, 4, 5]
                for grid_id in spatiotemporal_grids:
                    total += torch.abs(1 - grids[grid_id]).mean()
        if not torch.isfinite(total).all():
            print('CP04: Nan Found')
            return torch.tensor(0)
        else:
            return torch.as_tensor(total)


class DistortionLoss(Regularizer):
    def __init__(self, initial_value):
        super().__init__('distortion-loss', initial_value)

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        """
        Efficient O(N) realization of distortion loss.
        from https://github.com/sunset1995/torch_efficient_distloss/blob/main/torch_efficient_distloss/eff_distloss.py
        There are B rays each with N sampled points.
        """
        w = model_out['weights_list'][-1]
        rs: RaySamples = model_out['ray_samples_list'][-1]
        m = (rs.starts + rs.ends) / 2
        interval = rs.deltas

        loss_uni = (1 / 3) * (interval * w.pow(2)).sum(dim=-1).mean()
        wm = (w * m)
        w_cumsum = w.cumsum(dim=-1)
        wm_cumsum = wm.cumsum(dim=-1)
        loss_bi_0 = wm[..., 1:] * w_cumsum[..., :-1]
        loss_bi_1 = w[..., 1:] * wm_cumsum[..., :-1]
        loss_bi = 2 * (loss_bi_0 - loss_bi_1).sum(dim=-1).mean()
        loss = loss_bi + loss_uni

        if not torch.isfinite(loss):
            print('CP03: Nan Found')
            return torch.tensor(0)
        else:
            return loss


class L1TimePlanesRevgrid(Regularizer):
    def __init__(self, initial_value, what='rev_field'):

        super().__init__(f'l1-time-{what[:3]}', initial_value)
        self.what = what

    def _regularize(self, model: LowrankModel, **kwargs) -> torch.Tensor:
        # model.grids is 6 x [1, rank * F_dim, reso, reso]
        if self.what == 'rev_field' and model.disp_grid_used is not None:
            multi_res_grids = model.rev_grid.rev_disp_grids

        # elif self.what == 'rev_field' and model.disp_grid_used is None:
        #     multi_res_grids = model.field.grids

        else:
            raise NotImplementedError(self.what)

        total = 0.0
        for grids in multi_res_grids:
            if len(grids) == 3:
                continue
            else:
                # These are the spatiotemporal grids
                spatiotemporal_grids = [2, 4, 5]
            for grid_id in spatiotemporal_grids:
                total += torch.abs(1 - grids[grid_id]).mean()
        if not torch.isfinite(total):
            print('CP02: Nan Found')
            return torch.tensor(0)
        else :
            total




class Sparse_Flow_loss(Regularizer):
    def __init__(self, initial_value, what: str = 'optical_flow'):
        name = f'optical flow loss -{what[:3]}'
        super().__init__(name, initial_value)
        self.what = what

    def _regularize(self, model: LowrankModel, model_out, **kwargs) -> torch.Tensor:
        csv_points = model_out["csv_points"]
        # batch_size = 4096
        # portion_of_csv = 1 / 8          # CALL IT FROM THE DATASET

        p1 = model_out["t+10"]
        p2 = model_out["t-10"]

        x = model_out["x"]
        xt1, xt2 = x[-csv_points:-int(csv_points//2)], x[-int(csv_points//2):]
       #  = x_[:len(x_)//2], x_[len(x_)//2:]# xt1 belongs to points at t

        y = model_out["y"]
        yt1, yt2 = y[-csv_points:-int(csv_points//2)], y[-int(csv_points//2):]


        pt1 = p1[:len(p1)//2]  #  t+10
        pt2 = p2[len(p2)//2:]  #  t-10
        #t+10 optical loss
        #THE POINTS ARE ARRANGED IN DATA LOADING PART AS FIRST T+10 MOTION(POINTS ARE AT T IN CSV) AND OTHER HALF IS T-10 MOTION (POINTS AT T+10 IN CSV)
        # m1 = torch.column_stack([xt1, yt1])-pt1.to(xt1.device)
        # m2 = torch.column_stack([xt2, yt2])-pt2.to(xt1.device)
        m1 = pt1 - torch.column_stack([xt2, yt2])
        m2 = pt2 - torch.column_stack([xt1, yt1])
        m = torch.cat([m1, m2])
        # return torch.linalg.norm(m-v).mean()
        # loss_sm = torch.tensor(0).to(p1.device)
        loss_sm = torch.linalg.norm(m, dim=1).mean()
        if (loss_sm > 450).any():
            print('loss_sm above 100')
        if not torch.isfinite(loss_sm):
            print('CP01: Nan Found')
            return torch.tensor(0)
        else:
            return loss_sm

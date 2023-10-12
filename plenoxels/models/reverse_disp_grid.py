import itertools
import logging as log
from typing import Optional, Union, List, Dict, Sequence, Iterable, Collection, Callable

import torch
import torch.nn as nn
import tinycudann as tcnn

from plenoxels.ops.interpolation import grid_sample_wrapper
from plenoxels.raymarching.spatial_distortions import SpatialDistortion


def get_normalized_directions(directions):
    """SH encoding must be in the range [0, 1]

    Args:
        directions: batch of directions
    """
    return (directions + 1.0) / 2.0


def normalize_aabb(pts, aabb):
    return (pts - aabb[0]) * (2.0 / (aabb[1] - aabb[0])) - 1.0


def init_grid_param(
        grid_nd: int,
        in_dim: int,
        out_dim: int,
        reso: Sequence[int],
        a: float = 0.1,
        b: float = 0.5, only_time_plane=None):
    assert in_dim == len(reso), "Resolution must have same number of elements as input-dimension"
    has_time_planes = in_dim == 4
    assert grid_nd <= in_dim
    coo_combs = list(itertools.combinations(range(in_dim), grid_nd))
    grid_coefs = nn.ParameterList()
    if only_time_plane is None:
        for ci, coo_comb in enumerate(coo_combs):
            new_grid_coef = nn.Parameter(torch.empty(
                [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
            ))
            if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
                nn.init.ones_(new_grid_coef)
            else:
                nn.init.uniform_(new_grid_coef, a=a, b=b)
            grid_coefs.append(new_grid_coef)
# only for xt yt zt planes (time planes)
    else:
        for ci, coo_comb in enumerate(coo_combs):
            new_grid_coef = nn.Parameter(torch.empty(
                [1, out_dim] + [reso[cc] for cc in coo_comb[::-1]]
            ))
            if has_time_planes and 3 in coo_comb:  # Initialize time planes to 1
                nn.init.ones_(new_grid_coef)
                grid_coefs.append(new_grid_coef)

            else:
                nn.init.uniform_(new_grid_coef, a=a, b=b)

    return grid_coefs


def interpolate_ms_features(pts: torch.Tensor,
                            ms_grids: Collection[Iterable[nn.Module]],
                            grid_dimensions: int,
                            concat_features: bool,
                            num_levels: Optional[int],
                            in_dim = None,
                            time_planes_only = False) -> torch.Tensor:
    # ORIGINAL
    # if not time_planes_only and in_dim is None:
    #     coo_combs = list(itertools.combinations(
    #     range(pts.shape[-1]), grid_dimensions))
    #
    # else:
    coo_combs = list(itertools.combinations(range(pts.shape[-1]), grid_dimensions))

    if num_levels is None:
        num_levels = len(ms_grids)
    multi_scale_interp = [] if concat_features else 0.
    grid: nn.ParameterList
    for scale_id, grid in enumerate(ms_grids[:num_levels]):
        interp_space = 1.
        for ci, coo_comb in enumerate(coo_combs):
            # interpolate in plane
            feature_dim = grid[ci].shape[1]  # shape of grid[ci]: 1, out_dim, *reso
            interp_out_plane = (
                grid_sample_wrapper(grid[ci], pts[..., coo_comb])
                .view(-1, feature_dim)
            )
            # compute product over planes
            interp_space = interp_space * interp_out_plane

        # combine over scales
        if concat_features:
            multi_scale_interp.append(interp_space)
        else:
            multi_scale_interp = multi_scale_interp + interp_space

    if concat_features:
        multi_scale_interp = torch.cat(multi_scale_interp, dim=-1)
    return multi_scale_interp


class Rev_Disp_Field(nn.Module):
    def __init__(
        self,
        aabb,
        grid_config: Union[str, List[Dict]],
        concat_features_across_scales: bool,
        multiscale_res: Optional[Sequence[int]],
        # use_appearance_embedding: bool,
        # appearance_embedding_dim: int,
        spatial_distortion: Optional[SpatialDistortion],
        density_activation: Callable,
        linear_decoder: bool,
        # linear_decoder_layers: Optional[int],
        # num_images: Optional[int],
    ) -> None:
        super().__init__()

        self.aabb = nn.Parameter(aabb, requires_grad=False)
        self.spatial_distortion = spatial_distortion
        self.grid_config = grid_config

        self.multiscale_res_multipliers: List[int] = multiscale_res or [1]
        self.concat_features = concat_features_across_scales
        self.density_activation = density_activation
        self.linear_decoder = linear_decoder

        # 1. Init planes
        self.rev_disp_grids = nn.ModuleList()
        self.feature_dim = 0
        for res in self.multiscale_res_multipliers:
            # initialize coordinate grid
            config = self.grid_config[0].copy()
            # Resolution fix: multi-res only on spatial planes
            config["resolution"] = [
                r * res for r in config["resolution"][:3]
            ] + config["resolution"][3:]
            gp = init_grid_param(
                grid_nd=config["grid_dimensions"],
                in_dim=config["input_coordinate_dim"],
                out_dim=config["output_coordinate_dim"],
                reso=config["resolution"]
            )
#    shape[1] is out-dim - Concatenate over feature len for each scale
            if self.concat_features:
                self.feature_dim += gp[-1].shape[1]
            else:
                self.feature_dim = gp[-1].shape[1]
            self.rev_disp_grids.append(gp)
        self.rev_disp_net = tcnn.Network(
            n_input_dims=self.feature_dim,
            n_output_dims=3,
            network_config={
                "otype": "CutlassMLP",
                "activation": "None",
                "output_activation": "None",
                "n_neurons": 128,
                "n_hidden_layers": 0,
            },
        )
        log.info(f"Initialized reverse displacement grids: {self.rev_disp_grids}")

    def get_rev_disp(self, pts: torch.Tensor, timestamps: Optional[torch.Tensor] = None):
        pts = normalize_aabb(pts, self.aabb)
        n_rays, n_samples = pts.shape[:2]
        if timestamps is not None:
            timestamps = timestamps[:, None].expand(-1, n_samples)[..., None]  # [n_rays, n_samples, 1]
            pts = torch.cat((pts, timestamps), dim=-1)  # [n_rays, n_samples, 4]

        pts = pts.reshape(-1, pts.shape[-1])
        features = interpolate_ms_features(
            pts, ms_grids=self.rev_disp_grids,  # noqa
            grid_dimensions=self.grid_config[0]["grid_dimensions"],
            concat_features=self.concat_features, num_levels=None, in_dim=self.grid_config[0]["input_coordinate_dim"])
        if len(features) < 1:
            features = torch.zeros((0, 1)).to(features.device)
        disp = self.rev_disp_net(features).to(pts)

        return disp

    def forward(self,
                pts: torch.Tensor,
                timestamps: Optional[torch.Tensor] = None):

        disp = self.get_rev_disp(pts, timestamps)

        return disp

    def get_rev_disp_params(self):
        field_params = {k: v for k, v in self.rev_disp_grids.named_parameters(prefix="rev_disp_grids")}

        nn_params = [
                self.rev_disp_net.named_parameters(prefix="rev_disp_net")
                ]
        nn_params = {k: v for plist in nn_params for k, v in plist}

        return {
            "rev_nn": list(nn_params.values()),
            "rev_field": list(field_params.values()),
            # "other": list(other_params.values()),
        }

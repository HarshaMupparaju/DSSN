from typing import List, Sequence, Optional, Union, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from plenoxels.datasets.loding_csv import projection_without_near_shift, projection,projection_3d_poits,calc_o
from plenoxels.models.reverse_disp_grid import Rev_Disp_Field
from plenoxels.models.density_fields import KPlaneDensityField
from plenoxels.models.kplane_field import KPlaneField, disp_KPlaneField
from plenoxels.ops.activations import init_density_activation
from plenoxels.raymarching.ray_samplers import (
    UniformLinDispPiecewiseSampler, UniformSampler,
    ProposalNetworkSampler, RayBundle, RaySamples
)
from plenoxels.raymarching.spatial_distortions import SceneContraction, SpatialDistortion
from plenoxels.utils.timer import CudaTimer



class LowrankModel(nn.Module):
    def __init__(self,
                 disp_grid_config: Union[str, List[Dict]],
                 model_grid_config :Union[str, List[Dict]],
                 # boolean flags
                 is_ndc: bool,
                 is_contracted: bool,
                 aabb: torch.Tensor,
                 # Model arguments
                 multiscale_res: Sequence[int],
                 density_activation: Optional[str] = 'trunc_exp',
                 disp_concat_features_across_scales: bool = False,
                 model_concat_features_across_scales: bool = False,
                 linear_decoder: bool = True,
                 linear_decoder_layers: Optional[int] = 1,
                 # Spatial distortion
                 global_translation: Optional[torch.Tensor] = None,
                 global_scale: Optional[torch.Tensor] = None,
                 # proposal-sampling arguments
                 num_proposal_iterations: int = 1,
                 use_same_proposal_network: bool = False,
                 proposal_net_args_list: List[Dict] = None,
                 num_proposal_samples: Optional[Tuple[int]] = None,
                 num_samples: Optional[int] = None,
                 single_jitter: bool = False,
                 proposal_warmup: int = 5000,
                 proposal_update_every: int = 5,
                 use_proposal_weight_anneal: bool = True,
                 proposal_weights_anneal_max_num_iters: int = 1000,
                 proposal_weights_anneal_slope: float = 10.0,
                 # appearance embedding (phototourism)
                 use_appearance_embedding: bool = False,
                 appearance_embedding_dim: int = 0,
                 num_images: Optional[int] = None,
                 **kwargs,
                 ):
        super().__init__()
        # if isinstance(grid_config_disp, str):
        #     self.config_disp: List[Dict] = eval(grid_config_disp)
        #     self.grid_config_model = grid_config_model

        self.disp_grid_config: List[Dict] = disp_grid_config
        self.model_grid_config = model_grid_config
        self.multiscale_res = multiscale_res
        self.is_ndc = is_ndc
        self.is_contracted = is_contracted
        self.disp_concat_features_across_scales = disp_concat_features_across_scales
        self.model_concat_features_across_scales = model_concat_features_across_scales
        self.linear_decoder = linear_decoder
        self.linear_decoder_layers = linear_decoder_layers
        self.density_act = init_density_activation(density_activation)
        self.timer = CudaTimer(enabled=False)

        self.spatial_distortion: Optional[SpatialDistortion] = None
        if self.is_contracted:
            self.spatial_distortion = SceneContraction(
                order=float('inf'), global_scale=global_scale,
                global_translation=global_translation)
        self.disp_field = disp_KPlaneField(aabb,
                                           grid_config=self.disp_grid_config,
                                           concat_features_across_scales=self.disp_concat_features_across_scales,
                                           multiscale_res=self.multiscale_res,
                                           spatial_distortion=self.spatial_distortion,
                                           density_activation=self.density_act,
                                           linear_decoder=self.linear_decoder,
                                           )

        # add parameters of reverse grid to model

        self.rev_grid = Rev_Disp_Field(aabb,
                                       grid_config=self.disp_grid_config,
                                       concat_features_across_scales=self.disp_concat_features_across_scales,
                                       multiscale_res=self.multiscale_res,
                                       spatial_distortion=self.spatial_distortion,
                                       density_activation=self.density_act,
                                       linear_decoder=self.linear_decoder,
                                       )

        # self.motion_grid = Motion_Field(aabb, # changed on 24 aug
        #                                 grid_config=self.disp_grid_config,
        #                                 concat_features_across_scales=self.disp_concat_features_across_scales,
        #                                 multiscale_res=self.multiscale_res,
        #                                 spatial_distortion=self.spatial_distortion,
        #                                 density_activation=self.density_act,
        #                                 linear_decoder=self.linear_decoder,
        #                                 )
# changed k plane which doesn't include computing sigma from feature concatenation
# but from the grid itself with extra dimension
        self.field = KPlaneField(
            aabb,
            grid_config=self.model_grid_config,
            concat_features_across_scales=self.model_concat_features_across_scales,
            multiscale_res=self.multiscale_res,
            use_appearance_embedding=use_appearance_embedding,
            appearance_embedding_dim=appearance_embedding_dim,
            spatial_distortion=self.spatial_distortion,
            density_activation=self.density_act,
            linear_decoder=self.linear_decoder,
            linear_decoder_layers=self.linear_decoder_layers,
            num_images=num_images,
        )

        # Initialize proposal-sampling nets
        self.density_fns = []
        self.num_proposal_iterations = num_proposal_iterations
        self.proposal_net_args_list = proposal_net_args_list
        self.proposal_warmup = proposal_warmup
        self.proposal_update_every = proposal_update_every
        self.use_proposal_weight_anneal = use_proposal_weight_anneal
        self.proposal_weights_anneal_max_num_iters = proposal_weights_anneal_max_num_iters
        self.proposal_weights_anneal_slope = proposal_weights_anneal_slope
        self.proposal_networks = torch.nn.ModuleList()
        self.disp_grid_used = 1
        if use_same_proposal_network:
            assert len(self.proposal_net_args_list) == 1, "Only one proposal network is allowed."
            prop_net_args = self.proposal_net_args_list[0]
            network = KPlaneDensityField(
                aabb, spatial_distortion=self.spatial_distortion,
                density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args)
            self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for _ in range(self.num_proposal_iterations)])
        else:
            for i in range(self.num_proposal_iterations):
                prop_net_args = self.proposal_net_args_list[min(i, len(self.proposal_net_args_list) - 1)]
                network = KPlaneDensityField(
                    aabb, spatial_distortion=self.spatial_distortion,
                    density_activation=self.density_act, linear_decoder=self.linear_decoder, **prop_net_args,
                )
                self.proposal_networks.append(network)
            self.density_fns.extend([network.get_density for network in self.proposal_networks])

        update_schedule = lambda step: np.clip(
            np.interp(step, [0, self.proposal_warmup], [0, self.proposal_update_every]),
            1,
            self.proposal_update_every,
        )
        if self.is_contracted or self.is_ndc:
            initial_sampler = UniformLinDispPiecewiseSampler(single_jitter=single_jitter)
        else:
            initial_sampler = UniformSampler(single_jitter=single_jitter)
        self.proposal_sampler = ProposalNetworkSampler(
            num_nerf_samples_per_ray=num_samples,
            num_proposal_samples_per_ray=num_proposal_samples,
            num_proposal_network_iterations=self.num_proposal_iterations,
            single_jitter=single_jitter,
            update_sched=update_schedule,
            initial_sampler=initial_sampler
        )

    def step_before_iter(self, step):
        if self.use_proposal_weight_anneal:
            # anneal the weights of the proposal network before doing PDF sampling
            N = self.proposal_weights_anneal_max_num_iters
            # https://arxiv.org/pdf/2111.12077.pdf eq. 18
            train_frac = np.clip(step / N, 0, 1)
            bias = lambda x, b: (b * x) / ((b - 1) * x + 1)
            anneal = bias(train_frac, self.proposal_weights_anneal_slope)
            self.proposal_sampler.set_anneal(anneal)

    def step_after_iter(self, step):
        if self.use_proposal_weight_anneal:
            self.proposal_sampler.step_cb(step)

    @staticmethod
    def render_rgb(rgb: torch.Tensor, weights: torch.Tensor, bg_color: Optional[torch.Tensor]):
        comp_rgb = torch.sum(weights * rgb, dim=-2)
        accumulated_weight = torch.sum(weights, dim=-2)
        if bg_color is None:
            pass
        else:
            comp_rgb = comp_rgb + (1.0 - accumulated_weight) * bg_color
        return comp_rgb

    @staticmethod
    def render_point_alpha_decomp(pts: torch.Tensor, weights: torch.Tensor):
        comp_rgb = torch.sum(weights * pts, dim=-2)
        # accumulated_weight = torch.sum(weights, dim=-2)

        return comp_rgb

    @staticmethod
    def render_depth(weights: torch.Tensor, ray_samples: RaySamples, rays_d: torch.Tensor):
        steps = (ray_samples.starts + ray_samples.ends) / 2
        one_minus_transmittance = torch.sum(weights, dim=-2)
        depth = torch.sum(weights * steps, dim=-2) + one_minus_transmittance * rays_d[..., -1:]
        return depth

    @staticmethod
    def render_accumulation(weights: torch.Tensor):
        accumulation = torch.sum(weights, dim=-2)
        return accumulation

    def forward(self, data, rays_o, rays_d, bg_color, near_far: torch.Tensor, timestamps=None):

        # # ADD IMAGE PLANE POINTS AND POSES IN THE OUT FOR MOTION SUPERVISION

        """
        rays_o : [batch, 3]
        rays_d : [batch, 3]
        timestamps : [batch]
        near_far : [batch, 2]
        """
        # Fix shape for near-far
        w2c = data["w2c"]
        if data["csv_rand_points"] is not None:
            x = data["x"]
            y = data['y']

            csv_points = data["csv_rand_points"]
            if csv_points!= 512:
                m =1
                p=1
        else:
            x = 0
            y = 0
            csv_points = 0
        nears, fars = torch.split(near_far, [1, 1], dim=-1)
        if nears.shape[0] != rays_o.shape[0]:
            ones = torch.ones_like(rays_o[..., 0:1])
            nears = ones * nears
            fars = ones * fars

        ray_bundle = RayBundle(origins=rays_o, directions=rays_d, nears=nears, fars=fars)
        # Note: proposal sampler mustn't use timestamps (=camera-IDs) with appearance-embedding,
        #       since the appearance embedding should not affect density. We still pass them in the
        #       call below, but they will not be used as long as density-field resolutions
        #       are be 3D.
        ray_samples, weights_list, ray_samples_list = self.proposal_sampler.generate_ray_samples(
            ray_bundle, timestamps=timestamps, density_fns=self.density_fns)
        pts = ray_samples.get_positions()
        if not torch.isfinite(pts).all():
            print('CP021: Nan Found')
            a = 0
            b = 1

        # output with timestamps

        if self.disp_grid_used is not None:
            disp_out = self.disp_field(pts, timestamps=timestamps).reshape(pts.shape).to(torch.float32)  # output with timestamps
            if not torch.isfinite(disp_out).all():
                print('CP020: Nan Found')
                a = 0
                b = 1
            # disp_out = disp_out
            timestamps1 = timestamps[:, None, None].expand(disp_out.shape)  # [n_rays, n_samples,1]
            disp = torch.where(timestamps1 != -1, disp_out, 0)
            # feeding the displacement only for non-zero timestamps
            pi_prime = disp + pts  # Modify this
            field_out = self.field(pi_prime, ray_bundle.directions, timestamps=None)
        else:  # REMOVING THE DISPLACEMENT PART
            field_out = self.field(pts, timestamps=timestamps)
            disp_out = None
            raise RuntimeError('Field called on pi and not on pi_prime. Check')

        rev_disp = self.rev_grid(pi_prime, timestamps=timestamps).reshape(pts.shape)
        # rev_disp = rev_disp
        pi_dprime = pi_prime + rev_disp


        rgb, density = field_out["rgb"], field_out["density"]

        weights = ray_samples.get_weights(density)
        weights_list.append(weights)
        ray_samples_list.append(ray_samples)

        # point3d_rev_grid = self.render_point_alpha_decomp(pts=rev_disp, weights=weights)
        # pts_rendered = self.render_point_alpha_decomp(pts=pts, weights=weights) #  9sep

        p1_i_rev = self.rev_grid(pi_prime, timestamps=timestamps + (2*data["frame_diff_for_motion"] / 299)) # motion_points_10after   (changed on aug 24)
        p1_i_tprime = p1_i_rev.reshape(pts.shape) + pi_prime
        p2_i_rev = self.rev_grid(pi_prime, timestamps=timestamps - (2*data["frame_diff_for_motion"] / 299))  # motion_points_10before (changed on aug 24)
        p2_i_tprime = p2_i_rev.reshape(pts.shape) + pi_prime

        p1_tprime = self.render_point_alpha_decomp(pts=p1_i_tprime.reshape(pts.shape), weights=weights)
        p2_tprime = self.render_point_alpha_decomp(pts=p2_i_tprime.reshape(pts.shape), weights=weights)


        c = data["cam_indices_forcsv_pts"]
       # w1 = w2c
        if c is None:  # for test

            a1 =torch.tensor([23455,2423])
            a2=torch.tensor([523,523])
        else:  # for training

            a1 = projection(pts=p1_tprime[-data["csv_rand_points"]:, :, None], w2c=w2c, intrinsics=data["intrinsics"], near=1)
            a2 = projection(pts=p2_tprime[-data["csv_rand_points"]:, :, None], w2c=w2c, intrinsics=data["intrinsics"], near=1)
            # a1 = projection_without_near_shift(pts=pts[-data["csv_rand_points"]:, 0, :, None], intrinsics=data["intrinsics"], w2c=data["v"].to(pts.device))
            # a2 =projection_without_near_shift(pts=p2_tprime[-data["csv_rand_points"]:, :, None], intrinsics=data["intrinsics"],w2c=w2c)
            # a1 = wihtout_ndc_projection(pts=p1_tprime[-data["csv_rand_points"]:, :, None], w2c=w2c, intrinsics=data["intrinsics"], c2w=data["u"]).to(torch.float32)
            # a2 = wihtout_ndc_projection(pts=p2_tprime[-data["csv_rand_points"]:, :, None], w2c=w2c, intrinsics=data["intrinsics"], c2w=data["u"]).to(torch.float32)
            # a1 = changing_to_pixel_loc(a=p1_tprime[-data["csv_rand_points"]:, :, None], intrinsics=data["intrinsics"])
            # a2 = changing_to_pixel_loc(a=p2_tprime[-data["csv_rand_points"]:, :, None], intrinsics=data["intrinsics"])

        rgb = self.render_rgb(rgb=rgb, weights=weights, bg_color=bg_color)
        depth = self.render_depth(weights=weights, ray_samples=ray_samples, rays_d=ray_bundle.directions)
        accumulation = self.render_accumulation(weights=weights)

        outputs = {
            "rgb": rgb,
            "accumulation": accumulation,
            "depth": depth,

            "canonical_time_points": disp,
            "pts": pts,

            "rev_disp": pi_dprime,
            "x": x,
            "y": y,
            f"t+{data['frame_diff_for_motion']}": a1,
            f"t-{data['frame_diff_for_motion']}": a2,
            "csv_points": csv_points
        }

        """
        POSES,CANONICAL VOLUME POINTS AT T==-1 ,3D POINTS FROM REVERSE GRID
        (FROM OBTAINED POINTS AT T==-1 TO GIVEN T REVERSE MAPPING),
        PROJECTED POINTS TO DIFFERENT CAMERA POSES,
        (.CSV FILES GIVING X1,Y1,X2,Y2)
        """

        # These use a lot of GPU memory, so we avoid storing them for eval.
        if self.training:
            outputs["weights_list"] = weights_list
            outputs["ray_samples_list"] = ray_samples_list
        for i in range(self.num_proposal_iterations):
            outputs[f"prop_depth_{i}"] = self.render_depth(
                weights=weights_list[i], ray_samples=ray_samples_list[i], rays_d=ray_bundle.directions)
        return outputs

    def get_params(self, lr: float):
        model_params = self.field.get_params()
        disp_model_params = self.disp_field.get_disp_params()
        rev_disp_param = self.rev_grid.get_rev_disp_params()
        pn_params = [pn.get_params() for pn in self.proposal_networks]
        field_params = model_params["field"] + rev_disp_param["rev_field"] + disp_model_params['field'] + [p for pnp in pn_params for p in pnp["field"]]
        nn_params = model_params["nn"] + rev_disp_param["rev_nn"] + disp_model_params["nn"] + [p for pnp in pn_params for p in pnp["nn"]]
        other_params = model_params["other"] + [p for pnp in pn_params for p in pnp["other"]]
        return [
            {"params": field_params, "lr": lr},
            {"params": nn_params, "lr": lr},
            {"params": other_params, "lr": lr},
           # {"params": motion_param['motion_nn'], "lr": lr}, # changed on 24 aug
           # {"params": motion_param['motion_field'], "lr": lr}# changed on 24 aug
        ]


def changing_to_pixel_loc(a, intrinsics):
    k = []

    b = normalise(a[..., 0,0] / a[..., 2,0])  # shape of a is (3,4096,3,1) first 3 is for no. of training videos
    c = normalise(a[..., 1,0] / a[..., 2,0])  # normalise wrt z
    d = ((b + 1) / 2) * intrinsics.width
    k.append(d)
    e = (c + 1) / 2 * intrinsics.height  # converting in range (0-1) and then to pixel height
    k.append(e)
    return torch.column_stack([d, e])


def world_to_pixel(pts, w2c, intrinsics):
    """
    a = shape(n,3,1)
    w2c = w2c matrix with convention (x,y,z) size (n,3,4)
    intrinsics = has intrinsics height width and focal
    """
    opengl = True
    w2c_is_c2w =False
    # intrinsic should be of shape (n,3,3)
    intrinsic_matrix = torch.tensor(
        [intrinsics.focal_x, 0, intrinsics.center_x, 0, intrinsics.focal_y, intrinsics.center_y, 0, 0, 1]).type(torch.float32).reshape(3, 3)
    if opengl:
        p = torch.tensor([1, 0, 0, 0, -1, 0, 0, 0, -1]).type(torch.float32).reshape(3, 3)[None, :, :].repeat(pts.shape[0], 1, 1).to(pts.device)
        p1 = torch.tensor([-1, 0, 0, 0, -1, 0, 0, 0, 1]).type(torch.float32).reshape(3, 3)[None, :, :].repeat(pts.shape[0], 1, 1).to(pts.device)
        transformation = torch.matmul(p1*p1, w2c[:, :, :3])  # making it in convention (x,-y,-z)
        transformation1 = torch.matmul(transformation, pts)
        transformation2 = torch.matmul(p1*p1, w2c[:, :, 3, None])  # making it in convention (x,-y,-z)
        #  transformation2 = w2c[]
        p_pts = torch.matmul(intrinsic_matrix[None, ...], transformation1+transformation2)   # shape(n,3,1)
        normlize_p_pts = p_pts[:, 0:2, 0]/p_pts[:, 2:3, 0]
        return normlize_p_pts
    if w2c_is_c2w:
        p = torch.tensor([1, 0, 0, 0, -1, 0, 0, 0, -1]).type(torch.float32).reshape(3, 3)[None, :, :].repeat(pts.shape[0], 1, 1).to(pts.device)
        transformation = torch.matmul(p, torch.transpose(w2c[:, :, :3], 1, 2))  # making it in convention (x,-y,-z)
        transformation1 = torch.matmul(transformation, pts-w2c[:, :, 3, None])
        # transformation2 =  w2c[:, :, 3, None].cpu()  # making it in convention (x,-y,-z)
        #  transformation2 = w2c[]
        p_pts = torch.matmul(intrinsic_matrix[None, ...], transformation1)   # shape(n,3,1)
        normlize_p_pts = p_pts[:, 0:2, 0]/p_pts[:, 2:3, 0]
        return normlize_p_pts
    else:
        p = torch.tensor([1, 0, 0, 0, -1, 0, 0, 0, -1]).type(torch.float).reshape(3, 3)[None, :, :].repeat(pts.shape[0],
                                                                                                          1, 1)
        transformation = w2c[:, :, :3]  # making it in convention (x,-y,-z)
        transformation1 = torch.matmul(transformation, pts)
        transformation2 = w2c[:, :, 3, None]  # making it in convention (x,-y,-z)
        #  transformation2 = w2c[]
        p_pts = torch.matmul(intrinsic_matrix[None, ...], transformation1 + transformation2)  # shape(n,3,1)
        normlize_p_pts = p_pts[:, 0:2, 0] / p_pts[:, 2:3, 0]
        return normlize_p_pts


def corect_projection(a, w2c, intrinsics):
    """
        a = shape(n,3,1)
        w2c = w2c matrix with convention (x,y,z) size (n,4,4)
        intrinsics = has intrinsics height width and focal
        """
    opengl = True
    # intrinsic should be of shape (n,3,3)
    near = 1
    intrinsic_matrix = torch.tensor(
        [intrinsics.focal_x, 0, intrinsics.center_x, 0, intrinsics.focal_y, intrinsics.center_y, 0, 0, 1]).type(
        torch.float).reshape(3, 3)
    if opengl:
        p = torch.tensor([1, 0, 0, 0, -1, 0, 0, 0, -1]).type(torch.float).reshape(3, 3)[None, :, :].repeat(a.shape[0], 1, 1)
        # p1 is mapping from ndc to world
        p1 = torch.tensor([-intrinsics.width/(2*intrinsics.focal_x), 0, 0, 0, -intrinsics.height/(2*intrinsics.focal_y), 0, 0, 0, 1]).type(torch.float).reshape(3, 3)[None, :, :].repeat(a.shape[0], 1, 1)

        transformation = torch.matmul(p, w2c[:, :3, :3].cpu())  # making it in convention (x,-y,-z)
        c2w = torch.linalg.inv(w2c).cpu()
        t1 = torch.matmul(p1, a.cpu()) + torch.tensor([0, 0, 2*near])[None, ..., None].repeat(a.shape[0], 1, 1)+ c2w[:, :3, 3, None]
        transformation1 = torch.matmul(transformation, t1)
        # transformation1 /= torch.linalg.norm(transformation1, dim=-2, keepdim=True)
        p_pts = torch.matmul(intrinsic_matrix[None, ...], transformation1)  # shape(n,3,1)

        normlize_p_pts = p_pts[:, 0:2, 0] / p_pts[:, 2:3, 0]
        # normlize_p_pts1 = p_pts1[:, 0:2, 0] / p_pts1[:, 2:3, 0]
        return normlize_p_pts

def wihtout_ndc_projection(pts,w2c,intrinsics,c2w):
    """
        a = shape(n,3,1)
        w2c = w2c matrix with convention (x,y,z) size (n,4,4)
        intrinsics = has intrinsics height width and focal
        c2w n,4,4
        """
    opengl = True
    # intrinsic should be of shape (n,3,3)
    near = 1
    intrinsic_matrix = torch.tensor(
        [intrinsics.focal_x, 0, intrinsics.center_x, 0, intrinsics.focal_y, intrinsics.center_y, 0, 0, 1]).type(
        torch.float).reshape(3, 3).to(pts.device)
    if opengl:
        p = (torch.tensor([1, 0, 0, 0, -1, 0, 0, 0, -1]).type(torch.float32).reshape(3, 3)[None, :, :]
             .repeat(pts.shape[0], 1, 1).to(pts.device))
        if not torch.isfinite(intrinsic_matrix).all():
            print('CP013: Nan Found')

        transformation = torch.matmul(p, w2c[:, :3, :3])  # making it in convention (x,-y,-z)#TODO check if p has to multipled on both side
        t1 = torch.matmul(transformation, pts - c2w[:, :3, 3, None])
        if not torch.isfinite(t1).all():
            print('CP012: Nan Found')

        p_pts = torch.matmul(intrinsic_matrix[None, ...], t1)  # projected points shape(n,3,1)
        if not torch.isfinite(p_pts).all():
            print('CP010: Nan Found')

        normlize_p_pts = p_pts[:, 0:2, 0] / (p_pts[:, 2:3, 0]+0.00001)
        if not torch.isfinite(normlize_p_pts).all():
            print('CP011: Nan Found')
            a = 1

        return normlize_p_pts

def normalise(b):
    return (b-b.min())/(b.max()-b.min())

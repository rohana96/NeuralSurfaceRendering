import math
from typing import List

import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):


        n_rays = ray_bundle.origins.shape[0]
        origins = ray_bundle.origins

        dirs = ray_bundle.directions.unsqueeze(1)
        dirs = dirs.repeat(1, self.n_pts_per_ray, 1)

        # TODO (1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        z_vals = torch.linspace(start=self.min_depth, end=self.max_depth, steps=self.n_pts_per_ray)
        z_vals = z_vals.unsqueeze(0).unsqueeze(-1)
        z_vals = z_vals.repeat(n_rays, 1, 3)
        z_vals = z_vals.to('cuda')
        
        # TODO (1.4): Sample points from z values
        sample_points = origins.unsqueeze(1) + dirs*z_vals

        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals,
        )

sampler_dict = {
    'stratified': StratifiedRaysampler
}
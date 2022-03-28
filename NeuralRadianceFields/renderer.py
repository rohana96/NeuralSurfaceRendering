import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class VolumeRenderer(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False

    def _compute_weights(
        self,
        deltas,
        rays_density: torch.Tensor,
        eps: float = 1e-10
    ):
        # TODO (1.5): Compute transmittance using the equation described in the README
        n_rays, n_points, _ = deltas.shape

        transmittance = torch.ones(n_rays, 1, device=deltas.device)
        weight = transmittance * (1. - torch.exp(-rays_density[:, 0] * deltas[:, 0]))

        transmittances = [transmittance]
        weights = [weight]

        for i in range(1, n_points):
            transmittance = transmittances[i-1] * torch.exp(-rays_density[:, i-1] * deltas[:, i-1])
            transmittances.append(transmittance)
            weight = transmittance * (1. - torch.exp(-rays_density[:, i] * deltas[:, i]))
            weights.append(weight)
            
        # TODO (1.5): Compute weight used for rendering from transmittance and density
        weights = torch.stack(weights).to(deltas.device)
        weights = weights.permute(1, 0, 2)
        return weights
    
    def _aggregate(
        self,
        weights: torch.Tensor,
        rays_feature: torch.Tensor
    ):
        # TODO (1.5): Aggregate (weighted sum of) features using weights
        n_rays, n_points, _ = weights.shape
        rays_feature = rays_feature.reshape(n_rays, n_points, rays_feature.shape[-1])
        feature = torch.sum(rays_feature*weights, dim = 1)
        return feature

    def forward(
        self,
        sampler,
        implicit_fn,
        ray_bundle,
    ):
        B = ray_bundle.shape[0]
        # Process the chunks of rays.
        chunk_outputs = []

        for chunk_start in range(0, B, self._chunk_size):
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start+self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            implicit_output = implicit_fn(cur_ray_bundle)


            density = implicit_output['density']
            feature = implicit_output['feature']

            # Compute length of each ray segment
            depth_values = cur_ray_bundle.sample_lengths[..., 0]
            deltas = torch.cat(
                (
                    depth_values[..., 1:] - depth_values[..., :-1],
                    1e10 * torch.ones_like(depth_values[..., :1]),
                ),
                dim=-1,
            )[..., None]

            # Compute aggregation weights
            weights = self._compute_weights(
                deltas.view(-1, n_pts, 1),
                density.view(-1, n_pts, 1)
            ) 

            # TODO (1.5): Render (color) features using weights
            feature = self._aggregate(weights, feature)

            # TODO (1.5): Render depth map
            depth = self._aggregate(weights, depth_values.unsqueeze(-1))


            # Return
            cur_out = {
                'feature': feature,
                'depth': depth,
            }

            chunk_outputs.append(cur_out)

        # Concatenate chunk outputs
        out = {
            k: torch.cat(
              [chunk_out[k] for chunk_out in chunk_outputs],
              dim=0
            ) for k in chunk_outputs[0].keys()
        }

        return out


renderer_dict = {
    'volume': VolumeRenderer
}


def test_compute_weights():
    rays_density = torch.rand(size = (3, 4, 1))
    deltas = torch.rand(size = (3, 4, 1))
    n_rays, n_points, _ = deltas.shape
    transmittance = torch.ones_like(rays_density)

    for j in range(1, n_points):
        transmittance[:, j] = transmittance[:, j-1] * torch.exp(-1.0 * deltas[:, j-1] * rays_density[:, j-1])
    weights = transmittance * (1 - torch.exp( -1.0 * rays_density * deltas))

if __name__ == "__main__":
    test_compute_weights()
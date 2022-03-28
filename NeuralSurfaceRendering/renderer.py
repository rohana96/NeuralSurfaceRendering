import torch

from typing import List, Optional, Tuple
from pytorch3d.renderer.cameras import CamerasBase
import pdb


# Volume renderer which integrates color and density along rays
# according to the equations defined in [Mildenhall et al. 2020]
class SphereTracingRenderer(torch.nn.Module):
    def __init__(
            self,
            cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self.near = cfg.near
        self.far = cfg.far
        self.max_iters = cfg.max_iters

    def sphere_tracing(
            self,
            implicit_fn,
            origins,  # Nx3
            directions,  # Nx3
    ):
        """
        Input:
            implicit_fn: a module that computes a SDF at a query point
            origins: N_rays X 3
            directions: N_rays X 3
        Output:
            points: N_rays X 3 points indicating ray-surface intersections. For rays that do not intersect the surface,
                    the point can be arbitrary.
            mask: N_rays X 1 (boolean tensor) denoting which of the input rays intersect the surface.
        """
        # TODO (Q1): Implement sphere tracing
        # 1) Iteratively update points and distance to the closest surface
        #   in order to compute intersection points of rays with the implicit surface
        # 2) Maintain a mask with the same batch dimension as the ray origins,
        #   indicating which points hit the surface, and which do not
        n_rays, _ = origins.shape
        points = torch.zeros_like(origins)
        mask = torch.zeros(size=(n_rays, 1))
        for i in range(n_rays):
            start = origins[i, :] + directions[i, :] * self.near
            end = origins[i, :] + directions[i, :]*self.far
            max_dist = self.far
            dir = directions[i, :]
            closest_point = self._get_closest_point(
                implicit_fn,
                start,
                max_dist,
                dir,
                self.max_iters,
            )
            if closest_point != end:
                points[i, :] = closest_point
                mask[i, :] = 1

        return points, mask

    def _get_closest_point(self, implicit_fn, start, max_dist, dir, max_iter, eps=1e-5):
        p = start
        it = 0
        t = 0
        while implicit_fn(p) > eps and torch.linalg.norm(p) < max_dist and it < max_iter:
            t += implicit_fn(p)
            p = start + t * dir
            it += 1
        return p

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
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start + self._chunk_size]
            points, mask = self.sphere_tracing(
                implicit_fn,
                cur_ray_bundle.origins,
                cur_ray_bundle.directions
            )
            mask = mask.repeat(1, 3)
            isect_points = points[mask].view(-1, 3)

            # Get color from implicit function with intersection points
            isect_color = implicit_fn.get_color(isect_points)

            # Return
            color = torch.zeros_like(cur_ray_bundle.origins)
            color[mask] = isect_color.view(-1)

            cur_out = {
                'color': color.view(-1, 3),
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


def sdf_to_density(signed_distance, alpha, beta):
    # TODO (Q3): Convert signed distance to density with alpha, beta parameters
    pass


class VolumeSDFRenderer(torch.nn.Module):
    def __init__(
            self,
            cfg
    ):
        super().__init__()

        self._chunk_size = cfg.chunk_size
        self._white_background = cfg.white_background if 'white_background' in cfg else False
        self.alpha = cfg.alpha
        self.beta = cfg.beta

    def _compute_weights(
            self,
            deltas,
            rays_density: torch.Tensor,
            eps: float = 1e-10
    ):
        # TODO (Q3): Copy code from VolumeRenderer._compute_weights
        pass

    def _aggregate(
            self,
            weights: torch.Tensor,
            rays_color: torch.Tensor
    ):
        # TODO (Q3): Copy code from VolumeRenderer._aggregate
        pass

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
            cur_ray_bundle = ray_bundle[chunk_start:chunk_start + self._chunk_size]

            # Sample points along the ray
            cur_ray_bundle = sampler(cur_ray_bundle)
            n_pts = cur_ray_bundle.sample_shape[1]

            # Call implicit function with sample points
            distance, color = implicit_fn.get_distance_color(cur_ray_bundle.sample_points)
            color = color.view(-1, n_pts, 3)
            density = None  # TODO (Q3): convert SDF to density

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

            # Compute color
            color = self._aggregate(
                weights,
                color.view(-1, n_pts, color.shape[-1])
            )

            # Return
            cur_out = {
                'color': color,
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
    'sphere_tracing': SphereTracingRenderer,
    'volume_sdf': VolumeSDFRenderer
}

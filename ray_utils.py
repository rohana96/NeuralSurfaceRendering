import math
from typing import List, NamedTuple

import torch
import torch.nn.functional as F
from pytorch3d.renderer.cameras import CamerasBase


# Convenience class wrapping several ray inputs:
#   1) Origins -- ray origins
#   2) Directions -- ray directions
#   3) Sample points -- sample points along ray direction from ray origin
#   4) Sample lengths -- distance of sample points from ray origin

class RayBundle(object):
    def __init__(
            self,
            origins,
            directions,
            sample_points,
            sample_lengths,
    ):
        self.origins = origins
        self.directions = directions
        self.sample_points = sample_points
        self.sample_lengths = sample_lengths

    def __getitem__(self, idx):
        return RayBundle(
            self.origins[idx],
            self.directions[idx],
            self.sample_points[idx],
            self.sample_lengths[idx],
        )

    @property
    def shape(self):
        return self.origins.shape[:-1]

    @property
    def sample_shape(self):
        return self.sample_points.shape[:-1]

    def reshape(self, *args):
        return RayBundle(
            self.origins.reshape(*args, 3),
            self.directions.reshape(*args, 3),
            self.sample_points.reshape(*args, self.sample_points.shape[-2], 3),
            self.sample_lengths.reshape(*args, self.sample_lengths.shape[-2], 1),
        )

    def view(self, *args):
        return RayBundle(
            self.origins.view(*args, 3),
            self.directions.view(*args, 3),
            self.sample_points.view(*args, self.sample_points.shape[-2], 3),
            self.sample_lengths.view(*args, self.sample_lengths.shape[-2], 1),
        )

    def _replace(self, **kwargs):
        for key in kwargs.keys():
            setattr(self, key, kwargs[key])

        return self


# Sample image colors from pixel values
def sample_images_at_xy(
        images: torch.Tensor,
        xy_grid: torch.Tensor,
):
    batch_size = images.shape[0]
    spatial_size = images.shape[1:-1]

    xy_grid = -xy_grid.view(batch_size, -1, 1, 2)

    images_sampled = torch.nn.functional.grid_sample(
        images.permute(0, 3, 1, 2),
        xy_grid,
        align_corners=True,
        mode="bilinear",
    )

    return images_sampled.permute(0, 2, 3, 1).view(-1, images.shape[-1])


# Generate pixel coordinates from in NDC space (from [-1, 1])
def get_pixels_from_image(image_size, camera=None):

    H, W = image_size[0], image_size[1]

    # TODO (1.3): Generate pixel coordinates from [0, W] in x and [0, H] in y
    x = torch.linspace(start=0, end=W, steps=W)
    y = torch.linspace(start=0, end=H, steps=H)

    # TODO (1.3): Convert to the range [-1, 1] in both x and y
    grid_x = -1.0 * ((2 / W) * x - 1.0)
    grid_y = -1.0 * ((2 / H) * y - 1.0)
    # Create grid of coordinates
    xy_grid = torch.cartesian_prod(grid_y, grid_x)
    xy_grid = torch.flip(xy_grid, dims = [1] ) # swap [:, :, 0], [:, :, 1]
    return xy_grid.cuda()


# Random subsampling of pixels from an image
def get_random_pixels_from_image(n_pixels, image_size, camera=None):
    xy_grid = get_pixels_from_image(image_size, camera)

    # TODO (2.1): Random subsampling of pixel coordinates

    perm = torch.randperm(xy_grid.shape[0])
    idx = perm[:n_pixels]
    xy_grid_sub = xy_grid[idx]
    return xy_grid_sub.cuda()


# Get rays from pixel values
def get_rays_from_pixels(xy_grid, image_size=None, camera=None):

    # TODO (1.3): Map pixels to points on the image plane at Z=1
    xy_grid = xy_grid.cuda()
    ndc_points = xy_grid

    ndc_points = torch.cat(
        [
            ndc_points,
            torch.ones_like(ndc_points[..., -1:])
        ],
        dim=-1
    )

    # TODO (1.3): Use camera.unproject to get world space points on the image plane from NDC space points
    world_points = camera.unproject_points(ndc_points)

    # TODO (1.3): Get ray origins from camera center

    rays_o = camera.get_camera_center().expand(xy_grid.shape[0], -1)

    # TODO (1.3): Get normalized ray directions

    rays_d = world_points - rays_o
    rays_d = rays_d/(torch.linalg.norm(rays_d, dim=1).unsqueeze(1))
    
    # Create and return RayBundle
    return RayBundle(
        rays_o,
        rays_d,
        torch.zeros_like(rays_o).unsqueeze(1),
        torch.zeros_like(rays_o).unsqueeze(1),
    )


def test_get_pixels_from_image():
    print(get_pixels_from_image(image_size=(3, 3)))


def test_get_random_pixels_from_image():
    n_pixels = 10
    image_size = (24, 20)
    print(get_random_pixels_from_image(n_pixels=n_pixels, image_size=image_size))


def test_get_rays_from_pixels():
    image_size = (3, 3)
    xy_grid = get_pixels_from_image(image_size=image_size)
    print(get_rays_from_pixels(xy_grid=xy_grid))


if __name__ == '__main__':
    test_get_pixels_from_image()
    test_get_random_pixels_from_image()
    test_get_rays_from_pixels()

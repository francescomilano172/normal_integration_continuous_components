import numpy as np
import os
import trimesh

from pathlib import Path


def get_unique_filename(filepath):
    path = Path(filepath)
    counter = 1

    # If the file does not exist, return the original name.
    if not path.exists():
        return str(path)

    # Split stem and suffix (e.g., "file.txt" -> "file", ".txt").
    while True:
        new_name = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not new_name.exists():
            return str(new_name)
        counter += 1


def save_point_cloud_from_log_depth(
    log_depth, curr_subfolder, tau_a_vec, is_valid_pixel_mask, normal_image, suffix=""
):
    assert log_depth.ndim == 2
    H, W = log_depth.shape
    os.makedirs(curr_subfolder, exist_ok=True)
    print(os.path.abspath(curr_subfolder))
    reconstructed_point_image = np.zeros((H, W, 3))
    reconstructed_point_image[:] = np.nan
    reconstructed_point_image[..., 2] = np.exp(log_depth)
    reconstructed_point_image[..., 0] = (
        reconstructed_point_image[..., 2] * tau_a_vec[..., 0]
    )
    reconstructed_point_image[..., 1] = (
        reconstructed_point_image[..., 2] * tau_a_vec[..., 1]
    )

    curr_valid_mask = np.logical_and(
        np.logical_not(np.isnan(reconstructed_point_image).any(axis=-1)),
        is_valid_pixel_mask,
    )
    reconstructed_pc = trimesh.PointCloud(
        vertices=reconstructed_point_image[curr_valid_mask],
        colors=((normal_image + 1.0) * 0.5)[curr_valid_mask],
    )
    output_path = os.path.join(curr_subfolder, f"reconstructed_pc{suffix}.ply")
    output_path = get_unique_filename(output_path)
    _ = reconstructed_pc.export(output_path)

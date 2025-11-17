import argparse
import copy
import datetime
import matplotlib

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import torch
import torch_scatter
import yaml

from scipy.sparse import csr_matrix, spdiags
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import cg, LinearOperator

from surface_normal_integration.bini.bilateral_normal_integration import (
    bilateral_normal_integration,
)
from surface_normal_integration.bini.misc import sigmoid
from surface_normal_integration.datasets.diligent import DiLiGenTObject
from surface_normal_integration.utils.metrics import MADEComputer
from surface_normal_integration.utils.parsing import int_string_or_none, float_or_none

from normal_integration_continuous_components.datasets.custom import CustomObject
from normal_integration_continuous_components.data_structures.component_decomposition import (
    ComponentDecomposition,
)
from normal_integration_continuous_components.utils.io import (
    save_point_cloud_from_log_depth,
)
from normal_integration_continuous_components.utils.logging import (
    MADELogger,
    TimingLogger,
)


def make_diag_operator(D_data):
    """
    Returns a LinearOperator that performs multiplication by a diagonal matrix with
    entries given by D_data.
    """
    n = len(D_data)

    def matvec(x):
        return D_data * x

    def rmatvec(x):
        return D_data * x  # Same for symmetric diagonals.

    return LinearOperator((n, n), matvec=matvec, rmatvec=rmatvec, dtype=D_data.dtype)


plt.rcParams["figure.figsize"] = (30, 10)

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--obj_type", choices=["custom", "diligent"], default="diligent")
parser.add_argument("--obj_name", required=True, type=str)
parser.add_argument("--output_subfolder", required=True, type=str)
parser.add_argument("--allow_only_two_sided_intercomponent_edges", action="store_true")
parser.add_argument("--use_only_horizontal_vertical", action="store_true")
parser.add_argument("--cg_max_iter", type=int, default=5000)
parser.add_argument("--cg_tol", type=float, default=1.0e-3)
parser.add_argument(
    "--tol_relative_energy_change",
    type=float,
    default=1.0e-3,
    help=(
        "In later stages of the optimization, if the optimization energy between two "
        "subsequent iterations changes in *relative* terms by less than this "
        "threshold, meta-optimization is stopped early and a merging is performed."
    ),
)
parser.add_argument(
    "--use_log_depth_formulation",
    action="store_true",
    help=(
        "If this argument is passed, the log-depth formulation of previous methods is "
        "used instead of our formulation based on relative log scales."
    ),
)
parser.add_argument(
    "--freq_merging",
    type=int,
    default=5,
    help=(
        "Minimum frequency of component merging in number of iterations. If "
        "--use_log_depth_formulation is passed, it acts as the maximum number of "
        "iterations."
    ),
)
parser.add_argument(
    "--threshold_continuity_criterion_deg",
    type=float_or_none,
    required=True,
    help=(
        "Max angle (in degrees) that the normals at two neighboring pixels can have "
        "for the edge between the two pixels to be considered continuous. If None, no "
        "initial decomposition based on normal similarity is computed.",
    ),
)
parser.add_argument(
    "--outlier_reweighting_type",
    choices=["soft", "hard", "none"],
    default="soft",
    help=(
        "Type of outlier reweighting. If 'none', no outlier reweighting is applied. "
        "If 'hard', after the initial components have been made as globally continuous "
        "as possible, all residuals larger in magnitude than "
        "`threshold_noncontinuous_residual` are assigned hard weight 0 during later "
        "stages of the optimization. If 'soft', a soft sigmoid-based weight is "
        "multiplied to the original residual weight, so that -log10(abs(residual)) is "
        "linearly mapped to the argument of the sigmoid, with the mapping such that "
        "-log(threshold_noncontinuous_residual) is mapped to -4 (which maps to ~0.02 "
        "through the sigmoid function) and -log(threshold_surely_continuous_residual) "
        "to 4 (which maps to ~0.98 through the sigmoid function).",
    ),
)
parser.add_argument("--threshold_noncontinuous_residual", type=float, default=1.0e-3)
parser.add_argument(
    "--threshold_surely_continuous_residual",
    type=float,
    help="Should be provided i.f.f. `outlier_reweighting_type` is 'soft'.",
)
parser.add_argument(
    "--initial_filling_type",
    type=str,
    choices=["joint_optimize", "parallel_optimize"],
    default="parallel_optimize",
)
parser.add_argument(
    "--min_res_th",
    type=float,
    default=5.0e-8,
    help=(
        "For all instances where a connected-component computation is required, all "
        "input values below this threshold in magnitude are remapped to have magnitude "
        "equal to this value. This is necessary because the scipy.sparse algorithms "
        "interpret very small values as the absence of an edge.",
    ),
)
parser.add_argument(
    "--component_vis_log_freq",
    type=int_string_or_none,
    required=True,
    help=(
        "Frequency (in number of meta-optimization iterations) with which a "
        "connected-component image is saved to file. If 'only_after_merge', images are "
        "saved only after a merge operation is performed. If None, no saving is "
        "performed."
    ),
)
parser.add_argument("--compute_min_theoretical_made", action="store_true")
parser.add_argument("--log_intermediate_mades", action="store_true")
parser.add_argument("--log_timings", action="store_true")

args = parser.parse_args()

data_dir = args.data_dir
obj_type = args.obj_type
obj_name = args.obj_name
output_subfolder = args.output_subfolder
allow_only_two_sided_intercomponent_edges = (
    args.allow_only_two_sided_intercomponent_edges
)
use_only_horizontal_vertical = args.use_only_horizontal_vertical
cg_max_iter = args.cg_max_iter
cg_tol = args.cg_tol
tol_relative_energy_change = args.tol_relative_energy_change
use_log_depth_formulation = args.use_log_depth_formulation
freq_merging = args.freq_merging
if args.threshold_continuity_criterion_deg is not None:
    threshold_continuity_criterion = (
        args.threshold_continuity_criterion_deg * np.pi / 180.0
    )
else:
    threshold_continuity_criterion = None
outlier_reweighting_type = args.outlier_reweighting_type
threshold_noncontinuous_residual = args.threshold_noncontinuous_residual
threshold_surely_continuous_residual = args.threshold_surely_continuous_residual
initial_filling_type = args.initial_filling_type
min_res_th = args.min_res_th
component_vis_log_freq = args.component_vis_log_freq
compute_min_theoretical_made = args.compute_min_theoretical_made
log_intermediate_mades = args.log_intermediate_mades
log_timings = args.log_timings

if use_log_depth_formulation:
    assert threshold_continuity_criterion is None, (
        "The formulation based on log-depth is not compatible with use of a component "
        "decomposition."
    )
    print(
        "\033[93m\033[4mWARNING: Using original formulation based on log-depth rather "
        "than on relative log scales. \033[1mNOTE: No merging will be performed, and "
        "optimization will be terminated early instead, if as many iterations as the "
        "merging frequency are performed.\033[0m\033[0m\033[0m"
    )

assert threshold_noncontinuous_residual > 0
if outlier_reweighting_type == "soft":
    assert (
        threshold_surely_continuous_residual is not None
        and threshold_surely_continuous_residual < threshold_noncontinuous_residual
    )
else:
    if threshold_surely_continuous_residual is not None:
        print(
            "\033[93mNOTE: Ignoring argument `threshold_surely_continuous_residual` "
            "because `outlier_reweighting_type` is set to "
            f"'{outlier_reweighting_type}' rather than to 'soft'.\033[0m"
        )
    if (
        threshold_noncontinuous_residual is not None
    ) and outlier_reweighting_type == "none":
        print(
            "\033[93mNOTE: Ignoring argument `threshold_noncontinuous_residual` "
            "because `outlier_reweighting_type` is set to 'none' rather than to 'soft' "
            "or 'hard'.\033[0m"
        )

if isinstance(component_vis_log_freq, str):
    assert component_vis_log_freq == "only_after_merge"

assert min_res_th > 1.0e-8

if log_timings:
    assert not (compute_min_theoretical_made or log_intermediate_mades), (
        "Timing logging should only be performed when other loggings and intermediate "
        "MADE computation are disabled."
    )

# Load data.
if obj_type == "custom":
    diligent_object = CustomObject(
        data_dir=data_dir,
        obj_name=obj_name,
        normal_type="gt",
        num_shifts="8",
        lambda_m=0.5,
        k_sigmoid_lambda_m=None,
        w_b_to_a_outlier_th=1.1,
        num_ms=15,
    )
elif obj_type == "diligent":
    diligent_object = DiLiGenTObject(
        data_dir=data_dir,
        obj_name=obj_name,
        normal_type="gt",
        num_shifts="8",
        lambda_m=0.5,
        k_sigmoid_lambda_m=None,
        w_b_to_a_outlier_th=1.1,
        num_ms=15,
    )

H = diligent_object.H
W = diligent_object.W
K = diligent_object.K
obj_name = diligent_object.obj_name
v_where_is_valid = diligent_object.v_where_is_valid
u_where_is_valid = diligent_object.u_where_is_valid
is_valid_pixel_mask = diligent_object.is_valid_pixel_mask
w_b_to_a_full = diligent_object.w_b_to_a_full
channel_idx_to_du_dv_full = diligent_object.channel_idx_to_du_dv_full
du_dv_to_channel_idx_full = diligent_object.du_dv_to_channel_idx_full
n_a_vec = diligent_object.n_a_vec
n_b_vec_full = diligent_object.n_b_vec_full
tau_a_vec = diligent_object.tau_a_vec
if diligent_object.depth_gt is not None:
    log_depth_gt = np.log(diligent_object.depth_gt)
else:
    log_depth_gt = None

curr_subfolder = os.path.join(
    output_subfolder,
    datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{obj_name}",
)
os.makedirs(curr_subfolder)

if log_depth_gt is not None:
    made_computer = MADEComputer(
        is_valid_pixel_mask=is_valid_pixel_mask, log_depth_gt=log_depth_gt
    )
# Obtain weighting factor based on n_a.tau_a from BiNI.
nz_u_bini = bilateral_normal_integration(
    normal_map=n_a_vec,
    normal_mask=is_valid_pixel_mask,
    k=2,
    K=K,
    max_iter=None,
    tol=1e-4,
    force_all_iters=None,
    optimize=False,
)[1]


device = "cuda:0"

is_valid_pixel_mask_tensor = torch.from_numpy(is_valid_pixel_mask).to(device=device)
channel_idx_to_du_dv_full_tensor = torch.from_numpy(channel_idx_to_du_dv_full).to(
    device=device
)
nz_u_image_tensor = torch.full((H, W), fill_value=torch.nan, device=device)
nz_u_image_tensor[is_valid_pixel_mask_tensor] = torch.from_numpy(nz_u_bini).to(
    dtype=torch.float32, device=device
)

if log_timings:
    TimingLogger.init(log_path=os.path.join(curr_subfolder, "timings_log.txt"))
component_decomposition = ComponentDecomposition(
    valid_pixel_mask=is_valid_pixel_mask_tensor,
    channel_idx_to_du_dv=channel_idx_to_du_dv_full_tensor,
    allow_only_two_sided_intercomponent_edges=allow_only_two_sided_intercomponent_edges,
    use_only_horizontal_vertical=use_only_horizontal_vertical,
    log_depth_init=None,
)
if log_timings:
    TimingLogger.log("Completed component initialization with per-pixel components")
w_b_to_a_full = torch.from_numpy(w_b_to_a_full).to(device=device)


if threshold_continuity_criterion is not None:
    ####################################################################################
    ################# Initial decomposition based on normal variation ##################
    ####################################################################################
    # Clipping is to avoid over-/underflows of the dot product due to numerical effects.
    arccos_n_a_dot_n_b = np.arccos(
        np.clip(np.einsum("ijk,ijks->ijs", n_a_vec, n_b_vec_full), a_min=-1, a_max=1)
    )
    continuity_measure = arccos_n_a_dot_n_b.copy()
    if np.any(
        np.isnan(
            continuity_measure[diligent_object.is_valid_and_valid_neighbor_mask_full]
        )
    ):
        print(
            "\033[93m\033[1mNOTE: There are some NaN pixels in the valid part of the "
            "continuity measure map.\033[0m\033[0m"
        )
    continuity_measure = torch.from_numpy(continuity_measure).to(device=device)
    continuity_measure[continuity_measure < 1.0e-5] = 1.0e-5

    curr_intercomponent_edges = component_decomposition.intercomponent_edges
    curr_intercomponent_pixel_idx_a = curr_intercomponent_edges["pixel_idx_a"]
    curr_intercomponent_pixel_idx_b = curr_intercomponent_edges["pixel_idx_b"]
    curr_component_idx_a = curr_intercomponent_edges["component_idx_a"]
    curr_component_idx_b = curr_intercomponent_edges["component_idx_b"]
    num_intercomponent_edges = len(curr_component_idx_a)
    v_a = component_decomposition.v_where_is_valid[curr_intercomponent_pixel_idx_a]
    u_a = component_decomposition.u_where_is_valid[curr_intercomponent_pixel_idx_a]
    v_b = component_decomposition.v_where_is_valid[curr_intercomponent_pixel_idx_b]
    u_b = component_decomposition.u_where_is_valid[curr_intercomponent_pixel_idx_b]
    shift_b_a = component_decomposition.du_dv_to_channel_idx_tensor(
        du=u_b - u_a, dv=v_b - v_a
    )

    local_mask_selectable = (
        continuity_measure[v_a, u_a, shift_b_a] <= threshold_continuity_criterion
    )
    num_selectable_edges = torch.count_nonzero(local_mask_selectable).item()
    component_idx_a_selectable_edges = curr_component_idx_a[local_mask_selectable]
    component_idx_b_selectable_edges = curr_component_idx_b[local_mask_selectable]
    num_connected_components = component_decomposition.num_connected_components

    meta_graph_matrix = scipy.sparse.csr_matrix(
        (
            np.ones((num_selectable_edges,)),
            (
                component_idx_a_selectable_edges.cpu().numpy(),
                component_idx_b_selectable_edges.cpu().numpy(),
            ),
        ),
        shape=(num_connected_components, num_connected_components),
    )
    (num_new_component_indices, new_component_indices) = connected_components(
        csgraph=meta_graph_matrix, directed=False, return_labels=True
    )
    component_decomposition.apply_new_decomposition(
        connected_components_indices=torch.from_numpy(new_component_indices).to(
            device=device, dtype=torch.int64
        )
    )

    if log_timings:
        TimingLogger.log("Computed initial decomposition")

    print(f"#components = {component_decomposition.num_connected_components}")

    ####################################################################################
    ############################### Fill each component ################################
    curr_nonintercomponent_edges = component_decomposition.nonintercomponent_edges
    curr_nonintercomponent_pixel_idx_a = curr_nonintercomponent_edges["pixel_idx_a"]
    curr_nonintercomponent_pixel_idx_b = curr_nonintercomponent_edges["pixel_idx_b"]
    (curr_nonintercomponent_component_idx_a) = (
        component_decomposition._pixel_idx_to_component_idx[
            curr_nonintercomponent_pixel_idx_a
        ]
    )
    (curr_nonintercomponent_component_idx_b) = (
        component_decomposition._pixel_idx_to_component_idx[
            curr_nonintercomponent_pixel_idx_b
        ]
    )
    assert torch.all(
        curr_nonintercomponent_component_idx_a == curr_nonintercomponent_component_idx_b
    )
    num_nonintercomponent_edges = len(curr_nonintercomponent_component_idx_a)
    v_a = component_decomposition.v_where_is_valid[curr_nonintercomponent_pixel_idx_a]
    u_a = component_decomposition.u_where_is_valid[curr_nonintercomponent_pixel_idx_a]
    v_b = component_decomposition.v_where_is_valid[curr_nonintercomponent_pixel_idx_b]
    u_b = component_decomposition.u_where_is_valid[curr_nonintercomponent_pixel_idx_b]
    shift_b_a = component_decomposition.du_dv_to_channel_idx_tensor(
        du=u_b - u_a, dv=v_b - v_a
    )
    v_mb = 2 * v_a - v_b
    u_mb = 2 * u_a - u_b
    if initial_filling_type == "parallel_optimize":
        ###### Try filling up each component with a separate optimization problem ######
        # - Step 1: Sort by component index.
        sorted_nonintercomponent_component_idx, sorted_idx = torch.sort(
            curr_nonintercomponent_component_idx_a
        )
        # - Step 2: Find where component IDs change.
        block_change = (
            torch.nonzero(
                sorted_nonintercomponent_component_idx[1:]
                != sorted_nonintercomponent_component_idx[:-1],
                as_tuple=True,
            )[0]
            + 1
        )
        # - Step 3: Split residual indices by change points.
        splits = torch.tensor_split(sorted_idx, block_change.cpu())

        all_optimization_matrices = []

        for curr_nonintercomponent_local_edge_indices in splits:
            curr_num_nonintercomponent_edges = len(
                curr_nonintercomponent_local_edge_indices
            )

            curr_all_values = np.concatenate(
                [
                    # log_scale_a terms.
                    np.ones((curr_num_nonintercomponent_edges,)),
                    # log_scale_b terms.
                    -np.ones((curr_num_nonintercomponent_edges,)),
                ]
            )
            curr_row_indices = np.concatenate(
                [
                    # log_scale_a terms.
                    np.arange(curr_num_nonintercomponent_edges),
                    # log_scale_b terms.
                    np.arange(curr_num_nonintercomponent_edges),
                ]
            )

            curr_pixel_indices_a = curr_nonintercomponent_pixel_idx_a[
                curr_nonintercomponent_local_edge_indices
            ]
            curr_pixel_indices_b = curr_nonintercomponent_pixel_idx_b[
                curr_nonintercomponent_local_edge_indices
            ]

            _, curr_pixel_indices_to_unique_indices = torch.unique(
                torch.concatenate([curr_pixel_indices_a, curr_pixel_indices_b]),
                return_inverse=True,
            )
            curr_unique_indices_a = curr_pixel_indices_to_unique_indices[
                : len(curr_pixel_indices_a)
            ]
            curr_unique_indices_b = curr_pixel_indices_to_unique_indices[
                len(curr_pixel_indices_a) :
            ]
            curr_num_pixels = curr_pixel_indices_to_unique_indices.max().item() + 1
            curr_unique_indices_to_pixel_indices = torch.zeros(
                (curr_num_pixels,), dtype=torch.int64, device=device
            )

            curr_unique_indices_to_pixel_indices[
                curr_pixel_indices_to_unique_indices
            ] = torch.concatenate([curr_pixel_indices_a, curr_pixel_indices_b])

            curr_col_indices = np.concatenate(
                [
                    # log_scale_a terms.
                    curr_unique_indices_a.cpu().numpy(),
                    # log_scale_b terms.
                    curr_unique_indices_b.cpu().numpy(),
                ]
            )

            curr_log_z_a = component_decomposition.pixel_idx_to_log_depth[
                curr_pixel_indices_a
            ]
            curr_log_z_b = component_decomposition.pixel_idx_to_log_depth[
                curr_pixel_indices_b
            ]
            curr_v_a = v_a[curr_nonintercomponent_local_edge_indices]
            curr_u_a = u_a[curr_nonintercomponent_local_edge_indices]
            curr_shift_b_a = shift_b_a[curr_nonintercomponent_local_edge_indices]
            curr_v_mb = v_mb[curr_nonintercomponent_local_edge_indices]
            curr_u_mb = u_mb[curr_nonintercomponent_local_edge_indices]
            # NOTE: Non-valid -b neighbors are assigned the same log depth as their
            # corresponding b neighbor by default. To only consider valid -b neighbors,
            # set flag `allow_only_two_sided_intercomponent_edges` to True when
            # instantiating `ComponentDecomposition`. Considering only valid -b
            # neighbors results in more iterations.
            curr_nonintercomponent_pixel_idx_mb = component_decomposition._pixel_to_idx[
                curr_v_mb, curr_u_mb
            ]
            curr_log_z_mb = curr_log_z_b.clone()
            mask_valid_mb = torch.logical_not(
                torch.isnan(curr_nonintercomponent_pixel_idx_mb)
            )
            curr_log_z_mb[mask_valid_mb] = (
                component_decomposition.pixel_idx_to_log_depth[
                    curr_nonintercomponent_pixel_idx_mb[mask_valid_mb].to(
                        dtype=torch.int64
                    )
                ]
            )
            curr_log_w_b_to_a = torch.log(
                w_b_to_a_full[curr_v_a, curr_u_a, curr_shift_b_a]
            )

            curr_b = -(curr_log_z_a - curr_log_z_b - curr_log_w_b_to_a).cpu().numpy()

            # Bilateral + n_a.tau_a-based weighting.
            curr_n_a_tau_a_weight = (
                (nz_u_image_tensor[curr_v_a, curr_u_a] ** 2).cpu().numpy()
            )
            curr_W_matrix = sigmoid(
                (
                    (curr_log_z_mb - curr_log_z_a) ** 2
                    - (curr_log_z_a - curr_log_z_b) ** 2
                )
                .cpu()
                .numpy(),
                k=2 * curr_n_a_tau_a_weight,
            )
            bilateral_weights = curr_W_matrix.copy()
            curr_W_matrix = bilateral_weights * curr_n_a_tau_a_weight

            curr_scaled_A = csr_matrix(
                (
                    curr_all_values * np.sqrt(curr_W_matrix[curr_row_indices]),
                    (curr_row_indices, curr_col_indices),
                ),
                shape=(curr_num_nonintercomponent_edges, curr_num_pixels),
            )
            curr_A_mat = curr_scaled_A.T @ curr_scaled_A
            curr_b_vec = curr_scaled_A.T @ (np.sqrt(curr_W_matrix) * curr_b)
            curr_D = make_diag_operator(
                D_data=1 / np.clip(curr_A_mat.diagonal(), 1e-5, None)
            )  # Jacob preconditioner
            all_optimization_matrices.append(
                (curr_A_mat, curr_b_vec, curr_D, curr_unique_indices_to_pixel_indices)
            )

        TimingLogger.log("Formed optimization matrices")

        from joblib import Parallel, delayed

        results = Parallel(n_jobs=4)(
            delayed(cg)(A, b, M=D, maxiter=cg_max_iter, rtol=cg_tol)
            for (A, b, D, _) in all_optimization_matrices
        )

        all_delta_log_zs = torch.zeros(
            (component_decomposition._num_valid_pixels,), device=device
        )
        for curr_result, curr_optimization_matrix in zip(
            results, all_optimization_matrices
        ):
            curr_delta_log_zs = curr_result[0]
            curr_pixel_indices = curr_optimization_matrix[-1]
            all_delta_log_zs[curr_pixel_indices] = torch.from_numpy(
                curr_delta_log_zs
            ).to(dtype=torch.float32, device=device)

        component_decomposition.pixel_idx_to_log_depth = all_delta_log_zs
    elif initial_filling_type == "joint_optimize":
        # Run a single instance of conjugate gradient, removing the intercomponent
        # edges.
        all_values = np.concatenate(
            [
                # log_scale_a terms.
                np.ones((num_nonintercomponent_edges,)),
                # log_scale_b terms.
                -np.ones((num_nonintercomponent_edges,)),
            ]
        )
        row_indices = np.concatenate(
            [
                # log_scale_a terms.
                np.arange(num_nonintercomponent_edges),
                # log_scale_b terms.
                np.arange(num_nonintercomponent_edges),
            ]
        )
        # - This is an extra step to exclude all one-pixel components, since these have
        #   no non-intercomponent edges. This can further reduce the complexity of the
        #   optimization.
        _, curr_pixel_indices_to_unique_indices = torch.unique(
            torch.concatenate(
                [curr_nonintercomponent_pixel_idx_a, curr_nonintercomponent_pixel_idx_b]
            ),
            return_inverse=True,
        )
        curr_unique_indices_a = curr_pixel_indices_to_unique_indices[
            : len(curr_nonintercomponent_pixel_idx_a)
        ]
        curr_unique_indices_b = curr_pixel_indices_to_unique_indices[
            len(curr_nonintercomponent_pixel_idx_a) :
        ]
        curr_num_pixels = curr_pixel_indices_to_unique_indices.max().item() + 1
        curr_unique_indices_to_pixel_indices = torch.zeros(
            (curr_num_pixels,), dtype=torch.int64, device=device
        )

        curr_unique_indices_to_pixel_indices[curr_pixel_indices_to_unique_indices] = (
            torch.concatenate(
                [curr_nonintercomponent_pixel_idx_a, curr_nonintercomponent_pixel_idx_b]
            )
        )
        col_indices = np.concatenate(
            [
                # log_scale_a terms.
                curr_unique_indices_a.cpu().numpy(),
                # log_scale_b terms.
                curr_unique_indices_b.cpu().numpy(),
            ]
        )
        A = csr_matrix(
            (all_values, (row_indices, col_indices)),
            shape=(num_nonintercomponent_edges, curr_num_pixels),
        )
        curr_log_z_a = component_decomposition.pixel_idx_to_log_depth[
            curr_nonintercomponent_pixel_idx_a
        ]
        curr_log_z_b = component_decomposition.pixel_idx_to_log_depth[
            curr_nonintercomponent_pixel_idx_b
        ]
        # NOTE: Non-valid -b neighbors are assigned the same log depth as their
        # corresponding b neighbor by default. To only consider valid -b neighbors, set
        # flag `allow_only_two_sided_intercomponent_edges` to True when instantiating
        # `ComponentDecomposition`. Considering only valid -b neighbors results in more
        # iterations.
        # curr_log_z_mb = torch.zeros_like(curr_log_z_a)
        curr_nonintercomponent_pixel_idx_mb = component_decomposition._pixel_to_idx[
            v_mb, u_mb
        ]
        curr_log_z_mb = curr_log_z_b.clone()
        mask_valid_mb = torch.logical_not(
            torch.isnan(curr_nonintercomponent_pixel_idx_mb)
        )
        curr_log_z_mb[mask_valid_mb] = component_decomposition.pixel_idx_to_log_depth[
            curr_nonintercomponent_pixel_idx_mb[mask_valid_mb].to(dtype=torch.int64)
        ]
        curr_log_w_b_to_a = torch.log(w_b_to_a_full[v_a, u_a, shift_b_a])

        b = -(curr_log_z_a - curr_log_z_b - curr_log_w_b_to_a).cpu().numpy()

        # Bilateral + n_a.tau_a-based weighting.
        curr_n_a_tau_a_weight = (nz_u_image_tensor[v_a, u_a] ** 2).cpu().numpy()
        W_matrix = sigmoid(
            ((curr_log_z_mb - curr_log_z_a) ** 2 - (curr_log_z_a - curr_log_z_b) ** 2)
            .cpu()
            .numpy(),
            k=2 * curr_n_a_tau_a_weight,
        )
        bilateral_weights = W_matrix.copy()
        W_matrix = bilateral_weights * curr_n_a_tau_a_weight

        A_mat = A.T @ A.multiply(W_matrix[:, None])
        b_vec = A.T @ (W_matrix * b)
        D = spdiags(
            1 / np.clip(A_mat.diagonal(), 1e-5, None),
            0,
            curr_num_pixels,
            curr_num_pixels,
            format="csr",
        )  # Jacob preconditioner.

        TimingLogger.log("Formed optimization matrices")

        # - Run conjugate gradient.
        all_delta_log_zs = torch.zeros(
            (component_decomposition._num_valid_pixels,), device=device
        )
        all_delta_log_zs[curr_unique_indices_to_pixel_indices] = torch.from_numpy(
            cg(A_mat, b_vec, M=D, maxiter=cg_max_iter, rtol=cg_tol * 0.1)[0]
        ).to(dtype=torch.float32, device=device)
        # Update the log depth using the output from conjugate gradient, for
        # non-one-pixel components.
        component_decomposition.pixel_idx_to_log_depth = all_delta_log_zs
    else:
        raise ValueError()

    if log_timings:
        TimingLogger.log("Filled initial decomposition")

num_connected_components = component_decomposition.num_connected_components

if compute_min_theoretical_made and log_depth_gt is not None:
    min_theoretical_made_logger = MADELogger(
        made_fixed_message="Min theoretical MADE",
        log_path=os.path.join(curr_subfolder, "min_theoretical_mades.txt"),
    )
    app_reconstructed_z = torch.exp(component_decomposition.pixel_idx_to_log_depth)
    depth_gt_tensor = torch.from_numpy(made_computer._depth_gt).to(device=device)[
        is_valid_pixel_mask_tensor
    ]
    ratios = depth_gt_tensor / app_reconstructed_z
    num_pixels_per_component = torch.empty((num_connected_components,), device=device)
    mades = torch.empty((num_connected_components,), device=device)
    for component_idx in range(num_connected_components):
        curr_mask = component_decomposition._pixel_idx_to_component_idx == component_idx
        curr_median = torch.median(ratios[curr_mask])
        mades[component_idx] = torch.mean(
            torch.abs(
                app_reconstructed_z[curr_mask] * curr_median
                - depth_gt_tensor[curr_mask]
            )
        )
        num_pixels_per_component[component_idx] = torch.count_nonzero(curr_mask)
    pixel_weighted_made_disconnected = torch.sum(
        mades * num_pixels_per_component
    ) / torch.sum(num_pixels_per_component)
    min_theoretical_made_logger.log(
        iter_idx=-1,
        effective_iter_idx=-1,
        made=pixel_weighted_made_disconnected,
        num_connected_components=num_connected_components,
    )
if log_intermediate_mades and log_depth_gt is not None:
    intermediate_made_logger = MADELogger(
        made_fixed_message="Current MADE after optimization",
        log_path=os.path.join(curr_subfolder, "intermediate_mades.txt"),
    )
    curr_log_depth_image = np.full((H, W), fill_value=np.nan)
    curr_log_depth_image[is_valid_pixel_mask] = (
        component_decomposition.pixel_idx_to_log_depth.cpu().numpy()
    )
    curr_made_before_optim = made_computer.compute_curr_made(
        log_z=curr_log_depth_image, return_scale=False
    )[0]

    intermediate_made_logger.log(
        iter_idx=-1,
        effective_iter_idx=-1,
        made=curr_made_before_optim,
        num_connected_components=component_decomposition.num_connected_components,
    )

########################################################################################
idx = 0
num_effective_iterations = 0
all_log_zs = []
energy_old = None

curr_alpha_times_beta_term = torch.zeros(
    (len(component_decomposition.intercomponent_edges["pixel_idx_a"])), device=device
)

while num_connected_components > 1:

    if component_vis_log_freq is not None and (
        (
            component_vis_log_freq == "only_after_merge"
            and (idx == 0 or idx % freq_merging == 1 or freq_merging == 1)
        )
        or (
            component_vis_log_freq != "only_after_merge"
            and idx % component_vis_log_freq == 0
        )
    ):
        component_indices_color_encoded = (
            component_decomposition.get_connected_components_image()
        )
        plt.imsave(
            os.path.join(
                curr_subfolder,
                f"component_image_iter_{idx:04d}_num_components_"
                f"{num_connected_components}.png",
            ),
            component_indices_color_encoded,
        )
        plt.close()

    curr_intercomponent_edges = component_decomposition.intercomponent_edges
    curr_intercomponent_pixel_idx_a = curr_intercomponent_edges["pixel_idx_a"]
    curr_intercomponent_pixel_idx_b = curr_intercomponent_edges["pixel_idx_b"]
    curr_component_idx_a = curr_intercomponent_edges["component_idx_a"]
    curr_component_idx_b = curr_intercomponent_edges["component_idx_b"]
    num_intercomponent_edges = len(curr_component_idx_a)
    v_a = component_decomposition.v_where_is_valid[curr_intercomponent_pixel_idx_a]
    u_a = component_decomposition.u_where_is_valid[curr_intercomponent_pixel_idx_a]
    v_b = component_decomposition.v_where_is_valid[curr_intercomponent_pixel_idx_b]
    u_b = component_decomposition.u_where_is_valid[curr_intercomponent_pixel_idx_b]
    shift_b_a = component_decomposition.du_dv_to_channel_idx_tensor(
        du=u_b - u_a, dv=v_b - v_a
    )
    v_mb = 2 * v_a - v_b
    u_mb = 2 * u_a - u_b
    curr_intercomponent_pixel_idx_mb = component_decomposition._pixel_to_idx[v_mb, u_mb]

    curr_log_z_a = component_decomposition.pixel_idx_to_log_depth[
        curr_intercomponent_pixel_idx_a
    ]
    curr_log_z_b = component_decomposition.pixel_idx_to_log_depth[
        curr_intercomponent_pixel_idx_b
    ]
    # NOTE: Non-valid -b neighbors are assigned the same log depth as their
    # corresponding b neighbors by default. To only consider valid -b neighbors, set
    # flag `allow_only_two_sided_intercomponent_edges` to True when instantiating
    # `ComponentDecomposition`. Considering only valid -b neighbors results in more
    # iterations.
    # curr_log_z_mb = torch.zeros_like(curr_log_z_a)
    curr_log_z_mb = curr_log_z_b.clone()
    mask_valid_mb = torch.logical_not(torch.isnan(curr_intercomponent_pixel_idx_mb))
    curr_log_z_mb[mask_valid_mb] = component_decomposition.pixel_idx_to_log_depth[
        curr_intercomponent_pixel_idx_mb[mask_valid_mb].to(dtype=torch.int64)
    ]
    curr_w_b_to_a = w_b_to_a_full[v_a, u_a, shift_b_a]

    # Optimize scales/log depth.
    # - Form sparse matrix for CG.
    all_values = np.concatenate(
        [
            # log_scale_a terms (or log_depth_a terms if `use_log_depth_formulation` is
            # True).
            np.ones((num_intercomponent_edges,)),
            # log_scale_b terms (or log_depth_b terms if `use_log_depth_formulation` is
            # True).
            -np.ones((num_intercomponent_edges,)),
        ]
    )
    row_indices = np.concatenate(
        [
            # log_scale_a terms (or log_depth_a terms if `use_log_depth_formulation` is
            # True).
            np.arange(num_intercomponent_edges),
            # log_scale_b terms (or log_depth_b terms if `use_log_depth_formulation` is
            # True).
            np.arange(num_intercomponent_edges),
        ]
    )
    col_indices = np.concatenate(
        [
            # log_scale_a terms (or log_depth_a terms if `use_log_depth_formulation` is
            # True).
            curr_component_idx_a.cpu().numpy(),
            # log_scale_b terms (or log_depth_b terms if `use_log_depth_formulation` is
            # True).
            curr_component_idx_b.cpu().numpy(),
        ]
    )
    A = csr_matrix(
        (all_values, (row_indices, col_indices)),
        shape=(num_intercomponent_edges, num_connected_components),
    )

    if use_log_depth_formulation:
        b = torch.log(curr_w_b_to_a + curr_alpha_times_beta_term).cpu().numpy()
    else:
        b = (
            -(
                curr_log_z_a
                - curr_log_z_b
                - torch.log(curr_w_b_to_a + curr_alpha_times_beta_term)
            )
            .cpu()
            .numpy()
        )

    # Uniform weighting is used immediately after a merge operation.
    if threshold_continuity_criterion is None:
        use_uniform_weighting_in_curr_iter = (
            idx > freq_merging and idx % freq_merging == 1
        )
        # Uniform weighting is used also in the first iteration in which the surface is
        # most globally continuous. When no initial component initialization is used,
        # this happens in iteration 0, since the log depth map is initialized to all
        # zeros.
        use_uniform_weighting_in_curr_iter = (
            use_uniform_weighting_in_curr_iter or idx == 0
        )
    else:
        # NOTE: When using an initial decomposition, the first intecomponent
        # optimization (idx == 0) is run only to "align" the components in the most
        # globally continuous way, hence uniform weighting instead of BiNI weighting is
        # applied.
        use_uniform_weighting_in_curr_iter = idx == 0 or (
            idx > freq_merging and idx % freq_merging == 1
        )
        # Uniform weighting is used also in the first iteration in which the surface is
        # most globally continuous. When an initial component initialization is used,
        # this happens in iteration 1, since in iteration 0 the components are "aligned"
        # so as to be most globally continuous.
        use_uniform_weighting_in_curr_iter = (
            use_uniform_weighting_in_curr_iter or idx == 1
        )
    if not use_uniform_weighting_in_curr_iter:
        # Bilateral + n_a.tau_a-based weighting.
        n_a_tau_a_weight = (nz_u_image_tensor[v_a, u_a] ** 2).cpu().numpy()
        W_matrix = sigmoid(
            ((curr_log_z_mb - curr_log_z_a) ** 2 - (curr_log_z_a - curr_log_z_b) ** 2)
            .cpu()
            .numpy(),
            k=2 * n_a_tau_a_weight,
        )
        curr_alpha = (
            np.exp((curr_log_z_a - curr_log_z_b).cpu().numpy())
            - curr_w_b_to_a.cpu().numpy()
        )
        bilateral_weights = W_matrix.copy()
        W_matrix = bilateral_weights * n_a_tau_a_weight
    else:
        # Uniform weighting.
        W_matrix = np.ones((len(curr_log_z_a),))
    if idx > 0 and outlier_reweighting_type != "none":
        curr_abs_residuals = np.abs(
            (
                curr_log_z_a
                - curr_log_z_b
                - torch.log(curr_w_b_to_a + curr_alpha_times_beta_term)
            )
            .cpu()
            .numpy()
        )
        if outlier_reweighting_type == "hard":
            outlier_mask = curr_abs_residuals > threshold_noncontinuous_residual
            W_matrix[outlier_mask] = 0.0
            num_outliers = np.count_nonzero(outlier_mask)
        elif outlier_reweighting_type == "soft":
            # Maps the -log(threshold_noncontinuous_residual) to -4 (which is mapped to
            # ~0.02 by the sigmoid function) and
            # -log(threshold_surely_continuous_residual) to 4 (which is mapped to ~0.98
            # by the sigmoid function).
            N = -np.log10(threshold_noncontinuous_residual)
            S = -np.log10(threshold_surely_continuous_residual)
            outlier_reweighting = sigmoid(
                4 / (S - N) * (2 * (-np.log10(curr_abs_residuals)) - (N + S))
            )
            W_matrix = W_matrix * outlier_reweighting
        else:
            assert ValueError("This error should have been caught earlier.")
    A_mat = A.T @ A.multiply(W_matrix[:, None])
    b_vec = A.T @ (W_matrix * b)
    D = spdiags(
        1 / np.clip(A_mat.diagonal(), 1e-5, None),
        0,
        num_connected_components,
        num_connected_components,
        format="csr",
    )  # Jacob preconditioner
    # - Run conjugate gradient.
    if use_log_depth_formulation:
        # NOTE: `component_idx_to_log_scale` is a misnomer when log-depth formulation is
        # used, and corresponds instead to the log depth at each valid pixel at the
        # current iteration (i.e., the optimization variables).
        component_idx_to_log_scale = (
            component_decomposition.pixel_idx_to_log_depth.cpu().numpy()
        )
    else:
        component_idx_to_log_scale = np.zeros((num_connected_components,))
    component_idx_to_log_scale, _ = cg(
        A_mat,
        b_vec,
        x0=component_idx_to_log_scale,
        M=D,
        maxiter=cg_max_iter,
        rtol=cg_tol,
    )

    # Update the log depth of the pixels in each component.
    if use_log_depth_formulation:
        component_decomposition.pixel_idx_to_log_depth = torch.from_numpy(
            component_idx_to_log_scale
        ).to(device=device)[component_decomposition._pixel_idx_to_component_idx]
    else:
        component_decomposition.pixel_idx_to_log_depth += torch.from_numpy(
            component_idx_to_log_scale
        ).to(device=device)[component_decomposition._pixel_idx_to_component_idx]

    energy = (A @ component_idx_to_log_scale - b).T @ (
        W_matrix * (A @ component_idx_to_log_scale - b)
    )
    if energy_old is not None:
        relative_energy_change = np.abs(energy - energy_old) / energy_old
        if relative_energy_change < tol_relative_energy_change:
            # Force merge.
            idx += freq_merging - (idx % freq_merging)

    energy_old = copy.deepcopy(energy)

    all_log_zs.append(component_decomposition.pixel_idx_to_log_depth.cpu().numpy())

    if log_timings:
        TimingLogger.log(
            f"Computed optimization iteration {idx} (effective iteration "
            f"{num_effective_iterations})"
        )

    if log_intermediate_mades and log_depth_gt is not None:
        curr_log_depth_image = np.full((H, W), fill_value=np.nan)
        curr_log_depth_image[is_valid_pixel_mask] = (
            component_decomposition.pixel_idx_to_log_depth.cpu().numpy()
        )
        curr_made_after_optim = made_computer.compute_curr_made(
            log_z=curr_log_depth_image, return_scale=False
        )[0]
        intermediate_made_logger.log(
            iter_idx=idx,
            effective_iter_idx=num_effective_iterations,
            made=curr_made_after_optim,
            num_connected_components=(component_decomposition.num_connected_components),
        )

    if idx % freq_merging == 0 and idx > 0:
        if use_log_depth_formulation:
            print(
                "\033[93mTerminating before merging because log-depth formulation is "
                "being used.\033[0m"
            )
            break
        ################################################################################
        ###############             From here, recomputation             ###############
        # Compute residuals.
        curr_log_z_a = component_decomposition.pixel_idx_to_log_depth[
            curr_intercomponent_pixel_idx_a
        ]
        curr_log_z_b = component_decomposition.pixel_idx_to_log_depth[
            curr_intercomponent_pixel_idx_b
        ]

        curr_residuals_intercomponent = (
            curr_log_z_a - curr_log_z_b - torch.log(w_b_to_a_full[v_a, u_a, shift_b_a])
        )
        # - This is important to avoid wrong processing when computing the connected
        #   components.
        mask_curr_residuals_to_clamp = (
            torch.abs(curr_residuals_intercomponent) < min_res_th
        )
        curr_residuals_intercomponent[mask_curr_residuals_to_clamp] = (
            min_res_th
            * torch.sign(curr_residuals_intercomponent)[mask_curr_residuals_to_clamp]
        )
        curr_residuals_intercomponent[curr_residuals_intercomponent == 0] = min_res_th
        curr_abs_residuals_intercomponent = torch.abs(curr_residuals_intercomponent)

        # Select only edges in between components that have the smallest residual for
        # the outgoing component.
        (
            min_abs_curr_residuals_per_component,
            argmin_abs_curr_residuals_per_component,
        ) = torch_scatter.scatter_min(
            src=curr_abs_residuals_intercomponent, index=curr_component_idx_a
        )
        mask_argmin = torch.zeros(
            (num_intercomponent_edges,), device=device, dtype=bool
        )
        mask_argmin[
            argmin_abs_curr_residuals_per_component[
                argmin_abs_curr_residuals_per_component != num_intercomponent_edges
            ]
        ] = True
        local_mask_selectable_edges = mask_argmin

        # Compute connected components.
        abs_residual_selectable_edges = curr_abs_residuals_intercomponent[
            local_mask_selectable_edges
        ]
        component_idx_a_selectable_edges = curr_component_idx_a[
            local_mask_selectable_edges
        ]
        component_idx_b_selectable_edges = curr_component_idx_b[
            local_mask_selectable_edges
        ]
        meta_graph_matrix = scipy.sparse.csr_matrix(
            (
                abs_residual_selectable_edges.cpu().numpy(),
                (
                    component_idx_a_selectable_edges.cpu().numpy(),
                    component_idx_b_selectable_edges.cpu().numpy(),
                ),
            ),
            shape=(num_connected_components, num_connected_components),
        )

        # Directly compute connected components based on minimum-residual edge for each
        # component.
        (num_new_component_indices, new_component_indices) = connected_components(
            csgraph=meta_graph_matrix, directed=False, return_labels=True
        )

        # Apply the decomposition after merging to the original decomposition.
        component_decomposition.apply_new_decomposition(
            connected_components_indices=torch.from_numpy(new_component_indices).to(
                device=device, dtype=torch.int64
            )
        )

        curr_alpha_times_beta_term = torch.zeros(
            (len(component_decomposition.intercomponent_edges["pixel_idx_a"])),
            device=device,
        )

        num_connected_components = component_decomposition.num_connected_components

        if compute_min_theoretical_made and log_depth_gt is not None:
            app_reconstructed_z = torch.exp(
                component_decomposition.pixel_idx_to_log_depth
            )
            depth_gt_tensor = torch.from_numpy(made_computer._depth_gt).to(
                device=device
            )[is_valid_pixel_mask_tensor]
            ratios = depth_gt_tensor / app_reconstructed_z
            num_pixels_per_component = torch.empty(
                (num_connected_components,), device=device
            )
            mades = torch.empty((num_connected_components,), device=device)
            for component_idx in range(num_connected_components):
                curr_mask = (
                    component_decomposition._pixel_idx_to_component_idx == component_idx
                )
                curr_median = torch.median(ratios[curr_mask])
                mades[component_idx] = torch.mean(
                    torch.abs(
                        app_reconstructed_z[curr_mask] * curr_median
                        - depth_gt_tensor[curr_mask]
                    )
                )
                num_pixels_per_component[component_idx] = torch.count_nonzero(curr_mask)
            pixel_weighted_made_disconnected = torch.sum(
                mades * num_pixels_per_component
            ) / torch.sum(num_pixels_per_component)
            min_theoretical_made_logger.log(
                iter_idx=idx,
                effective_iter_idx=num_effective_iterations,
                made=pixel_weighted_made_disconnected,
                num_connected_components=num_connected_components,
            )

        if log_timings:
            TimingLogger.log(
                f"Performed merging at iteration {idx} (effective iteration "
                f"{num_effective_iterations})"
            )
    idx += 1
    num_effective_iterations += 1

if component_vis_log_freq is not None:
    component_indices_color_encoded = (
        component_decomposition.get_connected_components_image()
    )
    plt.imsave(
        os.path.join(curr_subfolder, "component_image_iter_final.png"),
        component_indices_color_encoded,
    )
    plt.close()

# Compute and print the final MADE.
curr_log_depth_image = np.full((H, W), fill_value=np.nan)
curr_log_depth_image[is_valid_pixel_mask] = (
    component_decomposition.pixel_idx_to_log_depth.cpu().numpy()
)
print_str = (
    f"Total number of iterations = {idx} (number of effective iterations = "
    f"{num_effective_iterations})."
)
made_kwargs = {}
if log_depth_gt is not None:
    made, _, made_scale = made_computer.compute_curr_made(
        log_z=curr_log_depth_image, return_scale=True
    )
    if obj_type == "custom":
        # The scale of the ground-truth point cloud for custom objects is in general in
        # an unknown unit of measure (e.g., unlike DiLiGenT, it is not guaranteed to be
        # in mm).
        made = None
        curr_depth_image = np.exp(curr_log_depth_image)
        curr_scaled_depth_image = curr_depth_image * made_scale
        mean_relative_depth_error = np.nanmean(
            np.abs(
                (curr_scaled_depth_image - made_computer._depth_gt)
                / made_computer._depth_gt
            )
        )
        avg_z_gt = np.nanmean(made_computer._depth_gt)
        mean_depth_error_relative_to_average_depth = np.nanmean(
            np.abs((curr_scaled_depth_image - made_computer._depth_gt) / avg_z_gt)
        )
        print_str_prefix = (
            "Final mean relative depth error = "
            f"{mean_relative_depth_error*100.0:.2f}%, final mean depth error relative "
            f"to average depth = "
            f"{mean_depth_error_relative_to_average_depth*100.0:.2f}%. "
        )
        made_kwargs = {
            "mean_relative_depth_error": float(mean_relative_depth_error),
            "mean_depth_error_relative_to_average_depth": float(
                mean_depth_error_relative_to_average_depth
            ),
        }
    else:
        print_str_prefix = f"Final MADE = {made:.3f}. "
        made_kwargs = {"made": float(made)}
    print_str = print_str_prefix + print_str

print(print_str)

# Save point cloud.
save_point_cloud_from_log_depth(
    log_depth=curr_log_depth_image,
    curr_subfolder=curr_subfolder,
    tau_a_vec=tau_a_vec,
    is_valid_pixel_mask=is_valid_pixel_mask,
    normal_image=n_a_vec,
)
if log_depth_gt is not None:
    save_point_cloud_from_log_depth(
        log_depth=np.log(made_computer._depth_gt / made_scale),
        curr_subfolder=curr_subfolder,
        tau_a_vec=tau_a_vec,
        is_valid_pixel_mask=is_valid_pixel_mask,
        normal_image=n_a_vec,
        suffix="_gt",
    )

# Log parameters and results to file.
with open(os.path.join(curr_subfolder, "config.yaml"), "w") as f:
    yaml.dump(
        {
            **vars(args),
            **made_kwargs,
        },
        f,
    )

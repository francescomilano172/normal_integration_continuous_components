import cv2
import numpy as np
import os

from scipy.io import loadmat
from typing import Optional, Union

from surface_normal_integration.datasets.common import Object


class CustomObject(Object):
    def __init__(
        self,
        data_dir: str,
        obj_name: str,
        normal_type: str,
        num_shifts: str,
        lambda_m: Union[float, str],
        k_sigmoid_lambda_m: Optional[Union[float, int]],
        w_b_to_a_outlier_th: Optional[float],
        num_ms: Optional[int],
        min_value_normal_dot_tau: Optional[float] = 0.0,
        threshold_rel_change_abs_n_dot_tau: Optional[float] = None,
        bad_normals_correction_criterion: str = "NaN",
    ):
        super().__init__(
            min_value_normal_dot_tau=min_value_normal_dot_tau,
            threshold_rel_change_abs_n_dot_tau=threshold_rel_change_abs_n_dot_tau,
            bad_normals_correction_criterion=bad_normals_correction_criterion,
            num_shifts=num_shifts,
        )

        self._data_dir = data_dir
        self._obj_list = ["bedroom", "living_room", "seafloor", "wedding_cake"]

        if obj_name in self._obj_list:
            self._obj_name = obj_name
        else:
            raise ValueError(
                f"Invalid object '{obj_name}'. Valid values are: {self._obj_list}."
            )

        assert normal_type == "gt"
        self._normal_type = normal_type

        # Load object.
        print(f"\nProcessing {self._obj_name} ...")
        mask_path = os.path.join(self._data_dir, "normals", self._obj_name, "mask.png")
        K_path = os.path.join(self._data_dir, "normals", self._obj_name, "K.txt")
        depth_gt_path = os.path.join(
            self._data_dir, "depth_gt", f"{self._obj_name}_gt.mat"
        )

        ray_directions_path = os.path.join(
            self._data_dir, "ray_directions", f"{self._obj_name}.mat"
        )

        assert normal_type == "gt"
        normal_image = loadmat(
            os.path.join(self._data_dir, f"pmsData/{self._obj_name}PNG/Normal_gt.mat")
        )["Normal_gt"].astype(np.float32)

        try:
            mask = cv2.imread(os.path.join(mask_path), cv2.IMREAD_GRAYSCALE).astype(
                bool
            )
        except:
            mask = np.ones(normal_image.shape[:2], bool)

        mask[0] = False
        mask[-1] = False
        mask[:, 0] = False
        mask[:, -1] = False

        normal_image[np.logical_not(mask)] = np.nan

        normal_image = normal_image / np.linalg.norm(
            normal_image, axis=-1, keepdims=True
        )
        # Convert normals to the coordinate frame type used.
        normal_image[..., 1:] = -normal_image[..., 1:]

        H, W = normal_image.shape[:2]

        K = np.loadtxt(K_path)
        K_bini = K.copy()
        K[0, 0] = K_bini[1, 1]
        K[1, 1] = K_bini[0, 0]
        K[0, 2] = K_bini[1, 2]
        K[1, 2] = K_bini[0, 2]

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]
        del K_bini

        u_image, v_image = np.meshgrid(np.arange(W), np.arange(H))

        try:
            depth_gt = loadmat(depth_gt_path)["depth_gt"]
            assert depth_gt.shape == (H, W)
            # - Form point cloud.
            z = depth_gt
            x = (u_image - cx) * z / fx
            y = (v_image - cy) * z / fy
            point_map_gt = np.stack([x, y, z], axis=-1)
        except FileNotFoundError:
            print(
                f"\033[93mCould not find ground-truth depth at '{depth_gt_path}'."
                "\033[0m"
            )
            depth_gt = None
            point_map_gt = None
        try:
            ray_directions = loadmat(ray_directions_path)
            tan_bear_angle_x = ray_directions["ray_directions_x"]
            tan_bear_angle_y = ray_directions["ray_directions_y"]
            print(f"\033[94mLoaded ray directions from '{ray_directions_path}'.\033[0m")
        except FileNotFoundError:
            tan_bear_angle_x = None
            tan_bear_angle_y = None

        # Compute view direction vectors and neighboring quantities.
        self._process_data(
            normal_image=normal_image,
            mask=mask,
            point_map_gt=point_map_gt,
            K=K,
            tan_bear_angle_x=tan_bear_angle_x,
            tan_bear_angle_y=tan_bear_angle_y,
            lambda_m=lambda_m,
            k_sigmoid_lambda_m=k_sigmoid_lambda_m,
            w_b_to_a_outlier_th=w_b_to_a_outlier_th,
            num_ms=num_ms,
        )

    @property
    def obj_name(self):
        return self._obj_name

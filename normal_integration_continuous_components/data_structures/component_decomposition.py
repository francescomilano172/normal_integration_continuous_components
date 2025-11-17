import numpy as np
import torch

from typing import Optional

from normal_integration_continuous_components.utils.processing import (
    depth_neighbor_from_depth_tensor,
)


class ComponentDecomposition:
    def __init__(
        self,
        valid_pixel_mask: torch.Tensor,
        channel_idx_to_du_dv: torch.Tensor,
        allow_only_two_sided_intercomponent_edges: bool,
        use_only_horizontal_vertical: bool,
        log_depth_init: Optional[torch.Tensor] = None,
    ):
        assert valid_pixel_mask.ndim == 2 and valid_pixel_mask.dtype == torch.bool
        self._H, self._W = valid_pixel_mask.shape
        self._valid_pixel_mask = valid_pixel_mask.clone()
        self._device = self._valid_pixel_mask.device
        assert channel_idx_to_du_dv.ndim == 2 and channel_idx_to_du_dv.shape[-1] == 2
        self._C = channel_idx_to_du_dv.shape[0]
        self._channel_idx_to_du_dv = channel_idx_to_du_dv
        self._du_dv_to_channel_idx = {
            (du, dv): idx for idx, (du, dv) in enumerate(self._channel_idx_to_du_dv)
        }

        self._valid_pixel_mask_neighbor = depth_neighbor_from_depth_tensor(
            depth=self._valid_pixel_mask,
            H=self._H,
            W=self._W,
            channel_idx_to_du_dv=self._channel_idx_to_du_dv,
        )
        assert self._valid_pixel_mask_neighbor.shape == (self._H, self._W, self._C)
        (self._v_where_is_valid, self._u_where_is_valid) = torch.where(
            self._valid_pixel_mask
        )
        (
            self._v_where_is_valid_and_valid_neighbor,
            self._u_where_is_valid_and_valid_neighbor,
            self._shift_where_is_valid_and_valid_neighbor,
        ) = torch.where(
            torch.logical_and(
                self._valid_pixel_mask[..., None], self._valid_pixel_mask_neighbor
            )
        )
        all_du_dvs = self._channel_idx_to_du_dv[
            self._shift_where_is_valid_and_valid_neighbor
        ]

        if allow_only_two_sided_intercomponent_edges or use_only_horizontal_vertical:
            if allow_only_two_sided_intercomponent_edges:
                mask_two_sided_edges = self._valid_pixel_mask[
                    self._v_where_is_valid_and_valid_neighbor - all_du_dvs[..., 1],
                    self._u_where_is_valid_and_valid_neighbor - all_du_dvs[..., 0],
                ]
                print("\033[4m\033[1mNOTE:\033[0m Only using two-sided edges.\033[0m")
            else:
                mask_two_sided_edges = torch.ones(
                    (len(self._v_where_is_valid_and_valid_neighbor),),
                    dtype=torch.bool,
                    device=self._device,
                )

            if use_only_horizontal_vertical:
                mask_horizontal_vertical = torch.isin(
                    self._shift_where_is_valid_and_valid_neighbor,
                    torch.where(torch.any(self._channel_idx_to_du_dv == 0, dim=-1))[0],
                )
                print(
                    "\033[04m\033[1mNOTE:\033[0m Only using horizontal-vertical edges."
                    "\033[0m"
                )
            else:
                mask_horizontal_vertical = torch.ones(
                    (len(self._v_where_is_valid_and_valid_neighbor),),
                    dtype=torch.bool,
                    device=self._device,
                )
            mask_filtering = torch.logical_and(
                mask_two_sided_edges, mask_horizontal_vertical
            )

            (self._v_where_is_valid_and_valid_neighbor) = (
                self._v_where_is_valid_and_valid_neighbor[mask_filtering]
            )
            (self._u_where_is_valid_and_valid_neighbor) = (
                self._u_where_is_valid_and_valid_neighbor[mask_filtering]
            )
            (self._shift_where_is_valid_and_valid_neighbor) = (
                self._shift_where_is_valid_and_valid_neighbor[mask_filtering]
            )
            all_du_dvs = all_du_dvs[mask_filtering]

        self._all_v_primes = (
            self._v_where_is_valid_and_valid_neighbor + all_du_dvs[..., 1]
        )
        self._all_u_primes = (
            self._u_where_is_valid_and_valid_neighbor + all_du_dvs[..., 0]
        )

        self._num_valid_pixels = len(self._v_where_is_valid)
        # Log depth at the valid pixels.
        if log_depth_init is not None:
            assert log_depth_init.shape == (self._num_valid_pixels,)
            self.pixel_idx_to_log_depth = log_depth_init
        else:
            self.pixel_idx_to_log_depth = torch.zeros(
                (self._num_valid_pixels,), device=self._device
            )

        # Data structure mapping pixels to their unique ID.
        self._pixel_to_idx = torch.full(
            (self._H, self._W), fill_value=torch.nan, device=self._device
        )
        self._pixel_to_idx[self._valid_pixel_mask] = torch.arange(
            self._num_valid_pixels, dtype=self._pixel_to_idx.dtype, device=self._device
        )

        self._pixel_idx_to_component_idx = torch.arange(
            self._num_valid_pixels, dtype=torch.int64, device=self._device
        )
        self._intercomponent_edges = {
            "pixel_idx_a": self._pixel_to_idx[
                self._v_where_is_valid_and_valid_neighbor,
                self._u_where_is_valid_and_valid_neighbor,
            ].to(dtype=torch.int64),
            "pixel_idx_b": self._pixel_to_idx[
                self._all_v_primes, self._all_u_primes
            ].to(dtype=torch.int64),
        }

        self._nonintercomponent_edges = {
            key: torch.empty((0,), dtype=torch.int64, device=self._device)
            for key in ["pixel_idx_a", "pixel_idx_b"]
        }
        self._update_auxiliary_intercomponent_structures()

    def apply_new_decomposition(self, connected_components_indices: torch.Tensor):
        assert (
            torch.is_tensor(connected_components_indices)
            and connected_components_indices.dtype == torch.int64
        )
        # Update per-pixel component indices.
        self._pixel_idx_to_component_idx = connected_components_indices[
            self._pixel_idx_to_component_idx
        ]

        # Update set of intercomponent edges.
        curr_component_indices_intercomponent_a = self._pixel_idx_to_component_idx[
            self._intercomponent_edges["pixel_idx_a"]
        ]
        curr_component_indices_intercomponent_b = self._pixel_idx_to_component_idx[
            self._intercomponent_edges["pixel_idx_b"]
        ]
        local_mask_intercomponent_edges = (
            curr_component_indices_intercomponent_a
            != curr_component_indices_intercomponent_b
        )
        for key in ["pixel_idx_a", "pixel_idx_b"]:
            self._nonintercomponent_edges[key] = torch.cat(
                [
                    self._nonintercomponent_edges[key],
                    self._intercomponent_edges[key][
                        torch.logical_not(local_mask_intercomponent_edges)
                    ],
                ]
            )
            self._intercomponent_edges[key] = self._intercomponent_edges[key][
                local_mask_intercomponent_edges
            ]
        self._update_auxiliary_intercomponent_structures()

    def _update_auxiliary_intercomponent_structures(self):
        curr_component_idx_a = self._pixel_idx_to_component_idx[
            self._intercomponent_edges["pixel_idx_a"]
        ]
        curr_component_idx_b = self._pixel_idx_to_component_idx[
            self._intercomponent_edges["pixel_idx_b"]
        ]
        self._intercomponent_edges["component_idx_a"] = curr_component_idx_a
        self._intercomponent_edges["component_idx_b"] = curr_component_idx_b

    @property
    def H(self):
        return self._H

    @property
    def W(self):
        return self._W

    @property
    def C(self):
        return self._C

    @property
    def num_connected_components(self):
        return torch.max(self._pixel_idx_to_component_idx).item() + 1

    @property
    def v_where_is_valid(self):
        return self._v_where_is_valid

    @property
    def u_where_is_valid(self):
        return self._u_where_is_valid

    @property
    def intercomponent_edges(self):
        return self._intercomponent_edges

    @property
    def nonintercomponent_edges(self):
        return self._nonintercomponent_edges

    def du_dv_to_channel_idx_tensor(self, du, dv):
        dv_du_to_channel_idx_ = torch.full(
            (3, 3), fill_value=torch.nan, device=self._device
        )
        for (du_, dv_), channel_idx in self._du_dv_to_channel_idx.items():
            dv_du_to_channel_idx_[dv_ + 1, du_ + 1] = channel_idx

        res = dv_du_to_channel_idx_[dv + 1, du + 1]
        assert torch.count_nonzero(torch.isnan(res)) == 0
        return res.to(dtype=torch.int64)

    def get_connected_components_image(self):
        colors_vis = np.random.rand(self.num_connected_components, 3)
        component_indices_color_encoded = np.full(
            (self._H, self._W, 3), fill_value=np.nan
        )
        component_indices_color_encoded[self._valid_pixel_mask.cpu().numpy()] = (
            colors_vis[self._pixel_idx_to_component_idx.cpu().numpy()]
        )
        return component_indices_color_encoded

import torch


def depth_neighbor_from_depth_tensor(depth, H, W, channel_idx_to_du_dv):
    assert depth.shape == (H, W)
    assert depth.device == channel_idx_to_du_dv.device
    device = depth.device

    du = channel_idx_to_du_dv[..., 0]
    dv = channel_idx_to_du_dv[..., 1]

    assert channel_idx_to_du_dv.ndim == 2 and channel_idx_to_du_dv.shape[-1] == 2
    C = channel_idx_to_du_dv.shape[0]

    H_padded = H + 2
    W_padded = W + 2
    indices_H_padded = (
        torch.arange(H_padded).view(H_padded, 1, 1).expand(H_padded, W_padded, C)
    ).to(device=device)
    indices_W_padded = (
        torch.arange(W_padded).view(1, W_padded, 1).expand(H_padded, W_padded, C)
    ).to(device=device)
    indices_shifts = (torch.arange(C).view(1, 1, C).expand(H_padded, W_padded, C)).to(
        device=device
    )

    depth_neighbor = depth.unsqueeze(-1).expand(-1, -1, C)
    depth_neighbor = torch.nn.functional.pad(
        depth_neighbor, (0, 0, 1, 1, 1, 1), value=float("nan")
    )
    assert depth_neighbor.shape == (H_padded, W_padded, C)

    depth_neighbor = depth_neighbor[
        (indices_H_padded + dv) % H_padded,
        (indices_W_padded + du) % W_padded,
        indices_shifts,
    ]

    depth_neighbor = depth_neighbor[1:-1, 1:-1, :]
    assert depth_neighbor.shape == (H, W, C)
    return depth_neighbor

# lib/models/projector.py
import torch

def projection(points_3d, cam_params):
    """
    Weak perspective projection
    points_3d: [B, J, 3]
    cam_params: [B, 3] with [scale, tx, ty]
    """
    batch_size = points_3d.shape[0]
    device = points_3d.device

    scale = cam_params[:, 0].unsqueeze(1).unsqueeze(2)
    trans = cam_params[:, 1:].unsqueeze(1)

    projected = scale * points_3d[:, :, :2] + trans
    return projected

import torch

def rot6d_to_rotmat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix using Gram-Schmidt."""
    x = x.view(-1, 6)
    x1 = x[:, 0:3]
    x2 = x[:, 3:6]

    b1 = F.normalize(x1, dim=1)
    b2 = F.normalize(x2 - (b1 * x2).sum(dim=1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)

    return torch.stack((b1, b2, b3), dim=-1)  # [N, 3, 3]

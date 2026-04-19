import torch


def total_variation_loss(x):
    if x.dim() not in [3, 4]:
        raise ValueError('Input must have shape (C, H, W) or (B, C, H, W).')

    if x.dim() == 3:
        x = x.unsqueeze(0)

    loss_h = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
    loss_w = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
    return loss_h + loss_w

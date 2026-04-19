import torch

try:
    import numpy as np
    from skimage.metrics import structural_similarity as skimage_ssim
except Exception:  # pragma: no cover - optional dependency fallback
    np = None
    skimage_ssim = None


def _validate_image_batch(original_batch, protected_batch):
    if original_batch.shape != protected_batch.shape:
        raise ValueError('Image batches must share the same shape for quality comparison.')
    if original_batch.dim() != 4:
        raise ValueError('Expected image batches with shape (B, C, H, W).')
    return original_batch.detach().cpu().float().clamp(0, 1), protected_batch.detach().cpu().float().clamp(0, 1)


def compute_l2_distance(original_batch, protected_batch):
    original_batch, protected_batch = _validate_image_batch(original_batch, protected_batch)
    diff = (protected_batch - original_batch).flatten(1)
    return torch.linalg.vector_norm(diff, ord=2, dim=1)


def compute_linf_distance(original_batch, protected_batch):
    original_batch, protected_batch = _validate_image_batch(original_batch, protected_batch)
    diff = (protected_batch - original_batch).abs().flatten(1)
    return diff.max(dim=1).values


def compute_psnr(original_batch, protected_batch):
    original_batch, protected_batch = _validate_image_batch(original_batch, protected_batch)
    mse = (protected_batch - original_batch).pow(2).flatten(1).mean(dim=1)
    return 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))


def _compute_ssim_with_skimage(original_batch, protected_batch):
    values = []
    for original_tensor, protected_tensor in zip(original_batch, protected_batch):
        original_image = original_tensor.permute(1, 2, 0).numpy()
        protected_image = protected_tensor.permute(1, 2, 0).numpy()
        values.append(
            skimage_ssim(
                original_image,
                protected_image,
                channel_axis=-1,
                data_range=1.0,
            )
        )
    return torch.tensor(values, dtype=torch.float32)


def _compute_ssim_with_torch(original_batch, protected_batch):
    kernel = torch.tensor([1.0, 4.0, 6.0, 4.0, 1.0], dtype=torch.float32)
    kernel = (kernel[:, None] * kernel[None, :]) / 256.0
    kernel = kernel.view(1, 1, 5, 5)

    def filter_image(image_batch):
        channels = image_batch.size(1)
        expanded_kernel = kernel.expand(channels, 1, 5, 5)
        return torch.nn.functional.conv2d(image_batch, expanded_kernel, padding=2, groups=channels)

    mu_x = filter_image(original_batch)
    mu_y = filter_image(protected_batch)
    sigma_x = filter_image(original_batch * original_batch) - mu_x.pow(2)
    sigma_y = filter_image(protected_batch * protected_batch) - mu_y.pow(2)
    sigma_xy = filter_image(original_batch * protected_batch) - mu_x * mu_y

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x.pow(2) + mu_y.pow(2) + c1) * (sigma_x + sigma_y + c2)
    ssim_map = numerator / denominator.clamp(min=1e-12)
    return ssim_map.flatten(1).mean(dim=1)


def compute_ssim(original_batch, protected_batch):
    original_batch, protected_batch = _validate_image_batch(original_batch, protected_batch)
    if skimage_ssim is not None and np is not None:
        return _compute_ssim_with_skimage(original_batch, protected_batch)
    return _compute_ssim_with_torch(original_batch, protected_batch)


def compute_batch_quality_metrics(original_batch, protected_batch):
    original_batch, protected_batch = _validate_image_batch(original_batch, protected_batch)
    return {
        'l2_distance': compute_l2_distance(original_batch, protected_batch),
        'linf_distance': compute_linf_distance(original_batch, protected_batch),
        'psnr': compute_psnr(original_batch, protected_batch),
        'ssim': compute_ssim(original_batch, protected_batch),
    }


def summarize_quality_records(records):
    if not records:
        return {}
    keys = ['l2_distance', 'linf_distance', 'psnr', 'ssim']
    return {f'avg_{key}': float(sum(record[key] for record in records) / len(records)) for key in keys}

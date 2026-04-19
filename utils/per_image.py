import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torchvision
from PIL import Image
from torchvision import transforms


IMAGE_EXTENSIONS = {'.bmp', '.jpeg', '.jpg', '.png', '.webp'}


def fix_random_seeds(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def build_attack_transform(image_size):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])



def parse_negative_texts(raw_text):
    if not raw_text:
        return []
    return [item.strip() for item in raw_text.split(',') if item.strip()]



def sanitize_filename(text):
    sanitized = re.sub(r'[^A-Za-z0-9._-]+', '_', str(text)).strip('._')
    return sanitized or 'sample'



def collect_input_images(input_image=None, input_dir=None):
    paths = []
    if input_image:
        image_path = Path(input_image)
        if not image_path.is_file():
            raise FileNotFoundError(f'Input image not found: {input_image}')
        paths.append(image_path)

    if input_dir:
        input_dir = Path(input_dir)
        if not input_dir.is_dir():
            raise FileNotFoundError(f'Input directory not found: {input_dir}')
        for path in sorted(input_dir.iterdir()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                paths.append(path)

    unique_paths = []
    seen = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            unique_paths.append(path)
            seen.add(resolved)
    return unique_paths



def load_image_tensor(image_path, transform):
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        return transform(image)



def tensor_to_unit_interval(image_tensor):
    image_tensor = image_tensor.detach().cpu().clamp(-1, 1)
    return (image_tensor * 0.5 + 0.5).clamp(0, 1)



def delta_to_visualization(delta, epsilon):
    delta = delta.detach().cpu()
    if epsilon > 0:
        scale = epsilon * 2
        return ((delta.clamp(-scale, scale) / scale) + 1) * 0.5

    delta = delta - delta.min()
    return delta / (delta.max() + 1e-8)



def save_tensor_image(image_tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torchvision.utils.save_image(image_tensor, str(path))



def save_compare_image(original_tensor, protected_tensor, path):
    original = tensor_to_unit_interval(original_tensor)
    protected = tensor_to_unit_interval(protected_tensor)
    grid = torchvision.utils.make_grid([original, protected], nrow=2, padding=2)
    save_tensor_image(grid, path)



def save_per_image_outputs(sample_dir, original_tensor, protected_tensor, delta_tensor, epsilon, metadata):
    sample_dir = Path(sample_dir)
    sample_dir.mkdir(parents=True, exist_ok=True)

    save_tensor_image(tensor_to_unit_interval(original_tensor), sample_dir / 'original.png')
    save_tensor_image(tensor_to_unit_interval(protected_tensor), sample_dir / 'protected.png')
    save_tensor_image(delta_to_visualization(delta_tensor, epsilon), sample_dir / 'delta_vis.png')
    save_compare_image(original_tensor, protected_tensor, sample_dir / 'compare.png')
    torch.save(delta_tensor.detach().cpu(), sample_dir / 'delta.pth')

    with open(sample_dir / 'log.json', 'w') as handle:
        json.dump(metadata, handle, indent=2)

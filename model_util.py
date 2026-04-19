import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def resize_perturbation(initialization, size):
    if not isinstance(initialization, torch.Tensor):
        raise TypeError("initialization must be a torch.Tensor")

    if initialization.dim() == 4:
        initialization = initialization.squeeze(0)
    if initialization.dim() != 3:
        raise ValueError("initialization must have shape (C, H, W) or (1, C, H, W)")

    if initialization.shape[-2:] == size:
        return initialization

    resized = F.interpolate(
        initialization.unsqueeze(0),
        size=size,
        mode='bilinear',
        align_corners=False,
    )
    return resized.squeeze(0)


def load_perturbation_checkpoint(ckpt_path, size, map_location='cpu'):
    initialization = torch.load(ckpt_path, map_location=map_location)
    if isinstance(initialization, dict):
        for key in ['trigger', 'delta', 'perturbation']:
            if key in initialization:
                initialization = initialization[key]
                break

    if not isinstance(initialization, torch.Tensor):
        raise ValueError("Checkpoint must contain a perturbation tensor.")

    return resize_perturbation(initialization, size)


def generate_points(w):
    return np.random.randint(0, w, size=(6, 2))

# Function to calculate the Euclidean distance between two points
def euclidean_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Function to drop rectangles in the image
def drop_blocks(image):
    c, w, h = image.shape
    points = generate_points(w)
    pairs = []

    # Repeat three times to find three pairs
    for _ in range(3):
        distances = {}
        
        # Calculate distances between each pair of points that haven't been paired yet
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if {i, j}.isdisjoint(set(sum(pairs, ()))):
                    dist = euclidean_distance(points[i], points[j])
                    distances[(i, j)] = dist
        
        # Find the pair of points with the smallest distance
        nearest_pair = min(distances, key=distances.get)
        pairs.append(nearest_pair)

    # Function to mask a block in the image
    def mask_block(image, start_point, end_point):
        x1, y1 = points[start_point]
        x2, y2 = points[end_point]
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)
        image[:, x_min:x_max+1, y_min:y_max+1] = 0

    # Apply masking to the three blocks
    for pair in pairs:
        mask_block(image, *pair)

    return image


class NoTargetDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, target):
        self.dataset = dataset
        self.indices = []
        self.target = target
        if target:
            if hasattr(self.dataset, 'targets') and len(self.dataset.targets) == len(self.dataset):
                for idx, y in enumerate(self.dataset.targets):
                    if y != target:
                        self.indices.append(int(idx))
            else:
                for idx, (X, y) in enumerate(self.dataset):
                    if y != target:
                        self.indices.append(int(idx))
        else:
            self.indices = [i for i in range(len(dataset))]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        return self.dataset[self.indices[item]]


class UniversalPerturbation(nn.Module):
    def __init__(self, size, epsilon=32/255, initialization=None, device='cuda'):
        super(UniversalPerturbation, self).__init__()
        # Initialize the trigger as a learnable parameter
        self.device = device
        if isinstance(initialization, torch.Tensor):
            self.trigger = nn.Parameter(initialization.clone().detach().to(self.device))
        else:
            self.trigger = nn.Parameter(torch.empty(size, requires_grad=True, device=self.device))
            nn.init.normal_(self.trigger, mean=0.0, std=0.01)
        self.epsilon = epsilon

    def forward(self, x):
        x = x.to(self.trigger.device)
        x_hat = x + torch.clamp(self.trigger, -self.epsilon * 2, self.epsilon * 2)
        return torch.clamp(x_hat, -1, 1)
  

class PerImagePerturbation(nn.Module):
    def __init__(self, reference_image, epsilon=32/255, initialization=None, device='cuda'):
        super(PerImagePerturbation, self).__init__()
        if reference_image.dim() != 3:
            raise ValueError("reference_image must have shape (C, H, W)")

        self.device = device
        self.epsilon = epsilon
        reference_image = reference_image.detach().to(self.device)

        if isinstance(initialization, torch.Tensor):
            delta = resize_perturbation(initialization, reference_image.shape[-2:])
            delta = delta.to(self.device, dtype=reference_image.dtype)
        else:
            delta = torch.empty_like(reference_image, device=self.device)
            nn.init.normal_(delta, mean=0.0, std=0.01)

        delta = delta.clamp(-self.epsilon * 2, self.epsilon * 2)
        self.delta = nn.Parameter(delta)

    def clamp_(self):
        with torch.no_grad():
            self.delta.clamp_(-self.epsilon * 2, self.epsilon * 2)

    def get_delta(self):
        return torch.clamp(self.delta, -self.epsilon * 2, self.epsilon * 2)

    def forward(self, x):
        x = x.to(self.delta.device)
        delta = self.get_delta()

        if x.dim() == 3:
            if x.shape != delta.shape:
                raise ValueError("Input image shape must match the learnable delta shape.")
            x_hat = x + delta
        elif x.dim() == 4:
            if x.shape[1:] != delta.shape:
                raise ValueError("Input image shape must match the learnable delta shape.")
            x_hat = x + delta.unsqueeze(0)
        else:
            raise ValueError("Input must have shape (C, H, W) or (B, C, H, W).")

        return torch.clamp(x_hat, -1, 1)


class BatchedPerImagePerturbation(nn.Module):
    def __init__(self, reference_images, epsilon=32/255, initialization=None, device='cuda'):
        super(BatchedPerImagePerturbation, self).__init__()
        if reference_images.dim() != 4:
            raise ValueError("reference_images must have shape (B, C, H, W)")

        self.device = device
        self.epsilon = epsilon
        reference_images = reference_images.detach().to(self.device)
        batch_size = reference_images.size(0)

        if isinstance(initialization, torch.Tensor):
            if initialization.dim() == 4 and initialization.size(0) == batch_size:
                delta = initialization.to(self.device, dtype=reference_images.dtype)
            else:
                base_delta = resize_perturbation(initialization, reference_images.shape[-2:])
                delta = base_delta.unsqueeze(0).repeat(batch_size, 1, 1, 1)
                delta = delta.to(self.device, dtype=reference_images.dtype)
        else:
            delta = torch.empty_like(reference_images, device=self.device)
            nn.init.normal_(delta, mean=0.0, std=0.01)

        delta = delta.clamp(-self.epsilon * 2, self.epsilon * 2)
        self.delta = nn.Parameter(delta)

    def clamp_(self):
        with torch.no_grad():
            self.delta.clamp_(-self.epsilon * 2, self.epsilon * 2)

    def get_delta(self):
        return torch.clamp(self.delta, -self.epsilon * 2, self.epsilon * 2)

    def forward(self, x):
        x = x.to(self.delta.device)
        if x.dim() != 4 or x.shape != self.delta.shape:
            raise ValueError("Input must have shape (B, C, H, W) matching the learnable batched delta.")

        return torch.clamp(x + self.get_delta(), -1, 1)


class BackdoorEval():
    def __init__(self, predictor, num_classes, device, target_class, target_only=True, top5=False):
        self.device = device
        self.num_classes = num_classes
        self.target_class = target_class
        self.target_only = target_only
        self.predict = predictor.to(self.device)
        self.predict.eval()
        self.top5 = top5

    def __call__(self, data_loader):
        total_predictions = torch.zeros(self.num_classes)
        top1_correct = 0
        top5_correct = 0
        total_samples = 0

        for i, (inputs, labels, *other_info) in enumerate(data_loader):
            inputs = inputs.to(self.device)
            outputs = self.predict(inputs)[:, :self.num_classes]
            preds = torch.argmax(outputs, dim=1).detach().cpu()
            total_predictions += torch.bincount(preds, minlength=self.num_classes)
            
            if self.top5:
                # Top-1 accuracy
                top1_correct += (preds == self.target_class).sum().item()
                
                # Top-5 accuracy
                top5_preds = torch.topk(outputs, 5, dim=1).indices
                top5_correct += sum([self.target_class in top5_preds[i].detach().cpu() for i in range(inputs.size(0))])
                
            total_samples += inputs.size(0)

        if self.target_only:
            # Calculate the percentage for the target class only
            total = total_predictions.sum()
            target_percentage = (total_predictions[self.target_class] / total) * 100
            if self.top5:
                top1_acc = (top1_correct / total_samples) * 100
                top5_acc = (top5_correct / total_samples) * 100
                return {
                    'Top-1 Accuracy': top1_acc,
                    'Top-5 Accuracy': top5_acc
                }
            return target_percentage.item()
        else:
            # Normalize counts to percentages for all classes
            total_predictions = (total_predictions / total_predictions.sum()) * 100
            if self.top5:
                top1_acc = (top1_correct / total_samples) * 100
                top5_acc = (top5_correct / total_samples) * 100
                return {
                    'Class Percentages': {f'Class {i}': p.item() for i, p in enumerate(total_predictions)},
                    'Top-1 Accuracy': top1_acc,
                    'Top-5 Accuracy': top5_acc
                }
            return {f'Class {i}': p.item() for i, p in enumerate(total_predictions)}


class TrainableAffineTransform(nn.Module):
    def __init__(self, batch_size=1, scale=0.05, flip=True, drop_blocks_aug=True):
        super(TrainableAffineTransform, self).__init__()
        # Initialize the parameters for the affine transformation with small random values
        theta = torch.eye(2, 3).unsqueeze(0).repeat(batch_size, 1, 1)

        # Apply small random transformations for rotation, translation, and scaling
        random_rotation = (torch.rand(batch_size) - 0.5) * 2 * scale  # Rotation angle between -5 and 5 degrees
        random_translation = (torch.rand(batch_size, 2) - 0.5) * 2 * scale  # Translation between -5% and 5% of image size
        random_scaling = 1 + (torch.rand(batch_size, 2) - 0.5) * 2 * scale  # Scaling between 0.95 and 1.05

        for i in range(batch_size):
            rotation_matrix = torch.tensor([
                [torch.cos(random_rotation[i]), -torch.sin(random_rotation[i]), random_translation[i, 0]],
                [torch.sin(random_rotation[i]), torch.cos(random_rotation[i]), random_translation[i, 1]]
            ], dtype=torch.float)

            scale_matrix = torch.diag(torch.cat([random_scaling[i], torch.tensor([1.0])])).unsqueeze(0)

            theta[i, :, :] = torch.mm(rotation_matrix, scale_matrix.squeeze())

        self.theta = nn.Parameter(theta, requires_grad=False)  # Set requires_grad=False as these are for augmentation
        self.flip = flip
        self.drop_blocks_aug = drop_blocks_aug

    def forward(self, x):
        batch_size = x.size(0)
        theta = self.theta[:batch_size].to(x.device)  # Ensure theta is the correct shape and on the correct device
        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)

        if self.flip:
            flip_indices = torch.randperm(batch_size)[:batch_size // 2]
            x[flip_indices] = torch.flip(x[flip_indices], [3])

        if self.drop_blocks_aug:
            # Drop two rectangular blocks
            for i in range(batch_size):
                x[i] = drop_blocks(x[i])

        return x

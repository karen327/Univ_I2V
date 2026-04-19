import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim

from data_util import GetDatasetMeta
from loss_function.clip_loss import CLIPLoss
from loss_function.regularization_loss import total_variation_loss
from model_util import BatchedPerImagePerturbation, TrainableAffineTransform, load_perturbation_checkpoint
from utils.per_image import (
    build_attack_transform,
    collect_input_images,
    fix_random_seeds,
    load_image_tensor,
    parse_negative_texts,
    sanitize_filename,
    save_compare_image,
    save_per_image_outputs,
)
from utils.text_templates import imagenet_templates


parser = argparse.ArgumentParser(description='Optimize an independent perturbation for each image.')
parser.add_argument('--mode', choices=['dataset', 'file'], required=True, help='Choose dataset mode or file mode.')
parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device used for optimization.')
parser.add_argument('--eps', type=int, default=32, help='Perturbation budget in 1/255, e.g., 32 for 32/255.')
parser.add_argument('--image_size', type=int, default=32, help='Input image resolution used during optimization.')
parser.add_argument('--steps', type=int, default=500, help='Optimization steps for each image.')
parser.add_argument('--lr', type=float, default=0.01, help='Learning rate for per-image perturbation optimization.')
parser.add_argument('--parallel_images', type=int, default=1, help='Number of images to optimize in parallel. Each image still keeps its own independent delta.')
parser.add_argument('--output_dir', default='./experiments/per_image', help='Base directory used to save per-image outputs. A run-specific subdirectory with key hyperparameters and a timestamp will be created automatically.')
parser.add_argument('--save_every', type=int, default=0, help='Save an intermediate compare image every N steps. Use 0 to disable.')
parser.add_argument('--surrogate', default='clip', choices=['clip', 'siglip', 'imagebind'], help='Text-image surrogate model used for optimization.')
parser.add_argument('--seed', type=int, default=0, help='Random seed.')
parser.add_argument('--pretrain_uap', default='', help='Optional UAP checkpoint used to warm-start each per-image delta.')
parser.add_argument('--data_path', default='/data/datasets', help='Root directory for datasets.')
parser.add_argument('--tgt_dataset', default=None, help='Target dataset name used in dataset mode or for file-mode templates/evaluation alignment.')
parser.add_argument('--target', type=int, default=None, help='Target class index used in dataset mode.')
parser.add_argument('--split', default='test', help='Dataset split for dataset mode: train, test, or val.')
parser.add_argument('--max_images', type=int, default=None, help='Maximum number of dataset images to optimize in dataset mode.')
parser.add_argument('--input_image', default='', help='Path to a single input image for file mode.')
parser.add_argument('--input_dir', default='', help='Directory of input images for file mode.')
parser.add_argument('--source_text', default='', help='Source text prompt for file mode.')
parser.add_argument('--target_text', default='', help='Target text prompt for file mode.')
parser.add_argument('--negative_texts', default='', help='Comma-separated negative text prompts for file mode.')
parser.add_argument('--aug_scale', type=float, default=0.25, help='Strength of differentiable affine augmentation during optimization. Lower values usually improve visual quality.')
parser.add_argument('--no_flip', action='store_true', help='Disable random horizontal flip augmentation in the per-image branch.')
parser.add_argument('--disable_block_drop', action='store_true', help='Disable random rectangular block dropping in the per-image branch.')
parser.add_argument('--tv_weight', type=float, default=0.0, help='Weight of total-variation regularization on the learned per-image delta.')
args = parser.parse_args()



def resolve_train_flag(split):
    split = split.lower()
    if split in ['train', 'training']:
        return True
    if split in ['test', 'testing', 'val', 'validation']:
        return False
    raise ValueError(f'Unsupported split: {split}. Expected train, test, or val.')



def validate_args(args):
    if args.steps <= 0:
        raise ValueError('--steps must be positive.')
    if args.lr <= 0:
        raise ValueError('--lr must be positive.')
    if args.image_size <= 0:
        raise ValueError('--image_size must be positive.')
    if args.parallel_images <= 0:
        raise ValueError('--parallel_images must be positive.')
    if args.save_every < 0:
        raise ValueError('--save_every must be non-negative.')
    if args.max_images is not None and args.max_images <= 0:
        raise ValueError('--max_images must be positive when provided.')
    if args.aug_scale < 0:
        raise ValueError('--aug_scale must be non-negative.')
    if args.tv_weight < 0:
        raise ValueError('--tv_weight must be non-negative.')

    if args.mode == 'dataset':
        if not args.tgt_dataset:
            raise ValueError('--tgt_dataset is required in dataset mode.')
        if args.target is None:
            raise ValueError('--target is required in dataset mode.')
    else:
        if not args.source_text or not args.target_text:
            raise ValueError('--source_text and --target_text are required in file mode.')
        if not args.input_image and not args.input_dir:
            raise ValueError('At least one of --input_image or --input_dir must be provided in file mode.')



def build_surrogate_loss(args, device):
    if args.surrogate != 'clip':
        raise NotImplementedError(f'surrogate={args.surrogate} is reserved for a later stage. Only clip is implemented now.')

    return CLIPLoss(
        device,
        lambda_direction=1,
        clip_model='ViT-B-32',
        pretrained='laion2b_s34b_b79k',
    )



def get_text_templates(args, data_meta=None):
    if data_meta is not None:
        return data_meta.get_template()
    if args.tgt_dataset:
        return GetDatasetMeta(args.data_path, args.tgt_dataset).get_template()
    return imagenet_templates



def build_dataset_records(args, transform):
    data_meta = GetDatasetMeta(args.data_path, args.tgt_dataset)
    label_dict = data_meta.get_dataset_label_names()
    if label_dict is None or args.target not in label_dict:
        raise ValueError(f'Target class {args.target} is not valid for dataset {args.tgt_dataset}.')

    dataset = data_meta.get_dataset(
        transform=transform,
        train=resolve_train_flag(args.split),
        download=True,
    )
    negative_texts = [label for idx, label in label_dict.items() if idx != args.target]
    target_text = label_dict[args.target]
    if not negative_texts:
        raise ValueError('Negative text set is empty. The target dataset must contain at least two classes.')

    def iterator():
        produced = 0
        for index in range(len(dataset)):
            sample = dataset[index]
            image, label = sample[0], sample[1]
            label = int(label.item()) if torch.is_tensor(label) else int(label)
            if label == args.target:
                continue

            yield {
                'sample_id': f'sample_{index:06d}_{sanitize_filename(label_dict[label])}',
                'image_tensor': image,
                'source_text': label_dict[label],
                'target_text': target_text,
                'negative_texts': negative_texts,
                'image_id': index,
                'source_class_index': label,
                'target_class_index': args.target,
                'tgt_dataset': args.tgt_dataset,
                'split': args.split,
                'input_path': None,
            }
            produced += 1
            if args.max_images is not None and produced >= args.max_images:
                break

    return iterator(), data_meta, list(label_dict.values())



def build_file_records(args, transform):
    input_paths = collect_input_images(args.input_image, args.input_dir)
    if not input_paths:
        raise ValueError('No input images found for file mode.')

    negative_texts = parse_negative_texts(args.negative_texts)
    if not negative_texts:
        negative_texts = [args.source_text]

    records = []
    for index, image_path in enumerate(input_paths):
        records.append({
            'sample_id': f'sample_{index:06d}_{sanitize_filename(Path(image_path).stem)}',
            'image_tensor': load_image_tensor(image_path, transform),
            'source_text': args.source_text,
            'target_text': args.target_text,
            'negative_texts': negative_texts,
            'image_id': Path(image_path).name,
            'source_class_index': None,
            'target_class_index': args.target,
            'tgt_dataset': args.tgt_dataset,
            'split': None,
            'input_path': str(image_path),
        })

    return records, None, sorted(set([args.source_text, args.target_text, *negative_texts]))



def build_run_output_dir(args):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    target_token = args.target if args.target is not None else args.target_text
    target_token = sanitize_filename(target_token if target_token is not None else 'none')

    run_parts = [
        f'eps_{args.eps}',
        f'img_{args.image_size}',
        f'target_{target_token}',
        f'aug_{sanitize_filename(args.aug_scale)}',
        f'tv_{sanitize_filename(args.tv_weight)}',
        f'par_{args.parallel_images}',
    ]
    if args.no_flip:
        run_parts.append('noflip')
    if args.disable_block_drop:
        run_parts.append('noblock')

    if args.mode == 'dataset':
        dataset_token = sanitize_filename(args.tgt_dataset)
        prefix = f'dataset_{dataset_token}'
    else:
        input_token = 'dir' if args.input_dir else 'image'
        prefix = f'file_{input_token}'

    run_name = '_'.join([prefix, *run_parts, timestamp])
    return Path(args.output_dir) / run_name



def chunk_records(records, chunk_size):
    batch = []
    for record in records:
        batch.append(record)
        if len(batch) == chunk_size:
            yield batch
            batch = []
    if batch:
        yield batch



def compute_similarity_metrics(loss_fn, original_tensor, protected_tensor, source_text, target_text, negative_texts):
    with torch.no_grad():
        src_features = loss_fn.get_image_features(original_tensor)
        protected_features = loss_fn.get_image_features(protected_tensor)
        edit_direction = protected_features - src_features
        edit_direction /= edit_direction.clone().norm(dim=-1, keepdim=True) + 1e-7

        target_direction = loss_fn.compute_text_direction([source_text], [target_text])
        negative_direction = loss_fn.compute_text_direction([source_text], negative_texts, broadcast=True)

        target_similarity = loss_fn.cosine_sim(target_direction, edit_direction).item()
        negative_similarity = loss_fn.cosine_sim(negative_direction, edit_direction.unsqueeze(-1)).squeeze(0)

    return {
        'target_similarity': float(target_similarity),
        'max_negative_similarity': float(negative_similarity.max().item()),
    }



def optimize_record_batch(records_batch, args, clip_loss_func, epsilon, warm_start):
    negative_texts = records_batch[0]['negative_texts']
    if any(record['negative_texts'] != negative_texts for record in records_batch[1:]):
        raise NotImplementedError('parallel batch optimization currently requires shared negative_texts across images in the same batch.')

    sample_dirs = [Path(args.output_dir) / record['sample_id'] for record in records_batch]
    if args.save_every > 0:
        for sample_dir in sample_dirs:
            (sample_dir / 'progress').mkdir(parents=True, exist_ok=True)

    originals = [record['image_tensor'].detach().cpu() for record in records_batch]
    image_batch = torch.stack([record['image_tensor'] for record in records_batch], dim=0).to(args.device).to(torch.float32)
    perturbation = BatchedPerImagePerturbation(
        image_batch,
        epsilon=epsilon,
        initialization=warm_start,
        device=args.device,
    )
    optimizer = optim.Adam(perturbation.parameters(), lr=args.lr, weight_decay=1e-5)

    source_texts = [record['source_text'] for record in records_batch]
    target_texts = [record['target_text'] for record in records_batch]
    history = []
    final_total_loss = None
    final_clip_loss = None
    final_tv_loss = None

    for step in range(1, args.steps + 1):
        protected = perturbation(image_batch)
        grad_transform = TrainableAffineTransform(
            image_batch.size(0),
            scale=args.aug_scale,
            flip=not args.no_flip,
            drop_blocks_aug=not args.disable_block_drop,
        )
        protected_aug = grad_transform(protected)
        clip_loss = clip_loss_func(
            image_batch,
            source_texts,
            protected_aug,
            target_texts,
            negative_texts,
        )
        tv_loss = total_variation_loss(perturbation.get_delta())
        loss = clip_loss + args.tv_weight * tv_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        perturbation.clamp_()

        final_total_loss = float(loss.item())
        final_clip_loss = float(clip_loss.item())
        final_tv_loss = float(tv_loss.item())

        if args.save_every > 0 and (step % args.save_every == 0 or step == args.steps):
            current_protected = perturbation(image_batch).detach().cpu()
            for idx, sample_dir in enumerate(sample_dirs):
                save_compare_image(originals[idx], current_protected[idx], sample_dir / 'progress' / f'step_{step:06d}.png')
            history.append({
                'step': step,
                'total_loss': final_total_loss,
                'clip_loss': final_clip_loss,
                'tv_loss': final_tv_loss,
            })

    final_protected_batch = perturbation(image_batch).detach().cpu()
    final_delta_batch = perturbation.get_delta().detach().cpu()

    for idx, record in enumerate(records_batch):
        metrics = compute_similarity_metrics(
            clip_loss_func,
            image_batch[idx:idx + 1],
            final_protected_batch[idx:idx + 1].to(args.device),
            record['source_text'],
            record['target_text'],
            record['negative_texts'],
        )

        metadata = {
            'image_id': record['image_id'],
            'input_path': record['input_path'],
            'mode': args.mode,
            'tgt_dataset': record['tgt_dataset'],
            'split': record['split'],
            'source_class_index': record['source_class_index'],
            'target_class_index': record['target_class_index'],
            'source_class_text': record['source_text'],
            'target_class_text': record['target_text'],
            'negative_texts': record['negative_texts'],
            'eps': args.eps,
            'image_size': args.image_size,
            'steps': args.steps,
            'lr': args.lr,
            'parallel_images': args.parallel_images,
            'surrogate': args.surrogate,
            'warm_start': bool(args.pretrain_uap),
            'aug_scale': args.aug_scale,
            'flip_aug': not args.no_flip,
            'block_drop_aug': not args.disable_block_drop,
            'tv_weight': args.tv_weight,
            'final_loss': final_total_loss,
            'final_clip_loss': final_clip_loss,
            'final_tv_loss': final_tv_loss,
            'optimization_batch_size': len(records_batch),
            **metrics,
        }
        if history:
            metadata['history'] = history

        save_per_image_outputs(
            sample_dirs[idx],
            originals[idx],
            final_protected_batch[idx],
            final_delta_batch[idx],
            epsilon,
            metadata,
        )
        print(
            f"[{record['sample_id']}] total_loss={final_total_loss:.4f}, "
            f"clip_loss={final_clip_loss:.4f}, tv_loss={final_tv_loss:.4f}, "
            f"target_similarity={metadata['target_similarity']:.4f}, "
            f"max_negative_similarity={metadata['max_negative_similarity']:.4f}"
        )



def main():
    validate_args(args)
    fix_random_seeds(args.seed)
    attack_transform = build_attack_transform(args.image_size)
    epsilon = args.eps / 255

    if args.mode == 'dataset':
        records, data_meta, text_vocab = build_dataset_records(args, attack_transform)
        templates = get_text_templates(args, data_meta)
    else:
        records, data_meta, text_vocab = build_file_records(args, attack_transform)
        templates = get_text_templates(args, data_meta)

    clip_loss_func = build_surrogate_loss(args, args.device)
    clip_loss_func.precompute_text_features(text_vocab, templates=templates)

    warm_start = None
    if args.pretrain_uap:
        warm_start = load_perturbation_checkpoint(args.pretrain_uap, (args.image_size, args.image_size), map_location='cpu')

    run_output_dir = build_run_output_dir(args)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir = str(run_output_dir)
    print(f'Saving outputs to {args.output_dir}')
    print(f'Optimizing up to {args.parallel_images} image(s) in parallel per GPU step.')

    total = 0
    for records_batch in chunk_records(records, args.parallel_images):
        optimize_record_batch(records_batch, args, clip_loss_func, epsilon, warm_start)
        total += len(records_batch)

    if total == 0:
        raise RuntimeError('No images were selected for optimization.')
    print(f'Finished optimizing {total} image(s).')


if __name__ == '__main__':
    main()

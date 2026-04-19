import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from data_util import GetDatasetMeta
from utils.quality_metrics import compute_batch_quality_metrics, summarize_quality_records


parser = argparse.ArgumentParser(description='Evaluate saved per-image protected samples with richer target-class diagnostics.')
parser.add_argument('--samples_path', required=True, help='A single sample directory or a directory containing multiple sample directories.')
parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu', help='Device used for evaluation.')
parser.add_argument('--data_path', default='/data/datasets', help='Root directory for datasets.')
parser.add_argument('--tgt_dataset', default=None, help='Target dataset name. If omitted, try reading it from each sample log.')
parser.add_argument('--target', type=int, default=None, help='Target class index. If omitted, try reading it from each sample log.')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size used for model evaluation.')
parser.add_argument('--skip_original_eval', action='store_true', help='Skip evaluating saved original.png files even when they are available.')
parser.add_argument('--skip_quality_eval', action='store_true', help='Skip lightweight image-quality metrics even when original.png is available.')
parser.add_argument('--verbose', action='store_true', help='Print per-sample diagnostics in addition to the aggregate summary.')
parser.add_argument('--output_json', default='', help='Optional path to save the evaluation summary JSON.')
args = parser.parse_args()

RAW_IMAGE_TRANSFORM = transforms.ToTensor()
QUALITY_KEYS = ['l2_distance', 'linf_distance', 'psnr', 'ssim']


def collect_sample_dirs(samples_path):
    samples_path = Path(samples_path)
    if not samples_path.exists():
        raise FileNotFoundError(f'Samples path not found: {samples_path}')
    if (samples_path / 'protected.png').is_file():
        return [samples_path]
    return sorted([path for path in samples_path.iterdir() if path.is_dir() and (path / 'protected.png').is_file()])


def load_metadata(sample_dir):
    log_path = sample_dir / 'log.json'
    if not log_path.is_file():
        return {}
    with open(log_path, 'r') as handle:
        return json.load(handle)


def resolve_shared_value(cli_value, metadata_list, key, value_name, arg_name):
    if cli_value is not None:
        return cli_value

    candidates = [metadata.get(key) for metadata in metadata_list if metadata.get(key) is not None]
    if not candidates:
        raise ValueError(f'{value_name} is required either via CLI or sample log.json metadata.')
    first_value = candidates[0]
    for candidate in candidates[1:]:
        if candidate != first_value:
            raise ValueError(f'Found inconsistent {value_name} values across sample logs. Please specify --{arg_name} explicitly.')
    return first_value


def build_eval_transform(dataset_meta):
    dataset_transform = dataset_meta.get_transformation()
    return transforms.Compose([transforms.ToTensor(), *dataset_transform.transforms])


def load_image_tensor(image_path, transform):
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        return transform(image)


def load_raw_image_tensor(image_path):
    return load_image_tensor(image_path, RAW_IMAGE_TRANSFORM)


def compute_topk_hit(logits, target_class, k):
    topk = min(k, logits.size(1))
    topk_indices = torch.topk(logits, topk, dim=1).indices
    return topk_indices.eq(target_class).any(dim=1)


def compute_target_rank(logits, target_class):
    target_logits = logits[:, target_class].unsqueeze(1)
    return (logits > target_logits).sum(dim=1) + 1


def evaluate_tensor_batch(model, image_batch, target_class, num_classes):
    with torch.no_grad():
        logits = model(image_batch)[:, :num_classes]
        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)
        top1_hit = predictions.eq(target_class)
        top5_hit = compute_topk_hit(logits, target_class, 5)
        target_rank = compute_target_rank(logits, target_class)
        target_prob = probabilities[:, target_class]
        target_logit = logits[:, target_class]

    return {
        'prediction': predictions.detach().cpu(),
        'top1_hit': top1_hit.detach().cpu(),
        'top5_hit': top5_hit.detach().cpu(),
        'target_rank': target_rank.detach().cpu(),
        'target_prob': target_prob.detach().cpu(),
        'target_logit': target_logit.detach().cpu(),
    }


def summarize_records(records, prefix):
    num_samples = len(records)
    summary = {
        f'{prefix}_num_samples': num_samples,
        f'{prefix}_target_top1_hits': int(sum(record[f'{prefix}_target_top1_hit'] for record in records)),
        f'{prefix}_target_top5_hits': int(sum(record[f'{prefix}_target_top5_hit'] for record in records)),
        f'{prefix}_target_top1_rate': float(sum(record[f'{prefix}_target_top1_hit'] for record in records) / num_samples),
        f'{prefix}_target_top5_rate': float(sum(record[f'{prefix}_target_top5_hit'] for record in records) / num_samples),
        f'avg_{prefix}_target_rank': float(sum(record[f'{prefix}_target_rank'] for record in records) / num_samples),
        f'avg_{prefix}_target_prob': float(sum(record[f'{prefix}_target_prob'] for record in records) / num_samples),
        f'avg_{prefix}_target_logit': float(sum(record[f'{prefix}_target_logit'] for record in records) / num_samples),
    }
    quality_records = [
        {key: record[key] for key in QUALITY_KEYS}
        for record in records
        if all(key in record for key in QUALITY_KEYS)
    ]
    summary.update(summarize_quality_records(quality_records))
    return summary


def build_source_group_key(source_class_index, source_class_text):
    if source_class_index is not None:
        return f'{int(source_class_index):04d}:{source_class_text}'
    if source_class_text:
        return str(source_class_text)
    return 'unknown'


def format_source_group_label(source_class_index, source_class_text):
    if source_class_index is not None:
        return f'{int(source_class_index)} ({source_class_text})'
    if source_class_text:
        return str(source_class_text)
    return 'unknown'


def summarize_source_groups(records, evaluate_original):
    grouped_records = {}
    for record in records:
        group_key = build_source_group_key(record.get('source_class_index'), record.get('source_class_text'))
        grouped_records.setdefault(group_key, []).append(record)

    source_group_summaries = []
    for group_key in sorted(grouped_records.keys()):
        group = grouped_records[group_key]
        first_record = group[0]
        group_summary = {
            'source_group_key': group_key,
            'source_class_index': first_record.get('source_class_index'),
            'source_class_text': first_record.get('source_class_text'),
            **summarize_records(group, 'protected'),
        }
        if evaluate_original:
            group_summary.update(summarize_records(group, 'original'))
            group_summary.update({
                'avg_target_rank_improvement': float(sum(record['target_rank_improvement'] for record in group) / len(group)),
                'avg_target_prob_increase': float(sum(record['target_prob_increase'] for record in group) / len(group)),
                'avg_target_logit_increase': float(sum(record['target_logit_increase'] for record in group) / len(group)),
            })
        source_group_summaries.append(group_summary)

    return source_group_summaries


def print_source_group_summaries(source_group_summaries, evaluate_original):
    if not source_group_summaries:
        return

    print('Source-class grouped summary:')
    for group_summary in source_group_summaries:
        label = format_source_group_label(group_summary.get('source_class_index'), group_summary.get('source_class_text'))
        message = (
            f"  [{label}] N={group_summary['protected_num_samples']}, "
            f"protected Top-1={group_summary['protected_target_top1_rate']:.4f}, "
            f"Top-5={group_summary['protected_target_top5_rate']:.4f}, "
            f"avg rank={group_summary['avg_protected_target_rank']:.2f}"
        )
        if evaluate_original:
            message += (
                f", rank improve={group_summary['avg_target_rank_improvement']:.2f}, "
                f"logit+={group_summary['avg_target_logit_increase']:.2f}"
            )
        if 'avg_ssim' in group_summary:
            message += (
                f", PSNR={group_summary['avg_psnr']:.2f}, "
                f"SSIM={group_summary['avg_ssim']:.4f}"
            )
        print(message)


def main():
    sample_dirs = collect_sample_dirs(args.samples_path)
    if not sample_dirs:
        raise RuntimeError('No protected sample directories were found.')

    metadata_list = [load_metadata(sample_dir) for sample_dir in sample_dirs]
    tgt_dataset = resolve_shared_value(args.tgt_dataset, metadata_list, 'tgt_dataset', 'target dataset', 'tgt_dataset')
    target_class = resolve_shared_value(args.target, metadata_list, 'target_class_index', 'target class index', 'target')

    dataset_meta = GetDatasetMeta(args.data_path, tgt_dataset)
    label_dict = dataset_meta.get_dataset_label_names()
    num_classes = len(label_dict)
    if target_class not in label_dict:
        raise ValueError(f'Target class {target_class} is not valid for dataset {tgt_dataset}.')

    model = dataset_meta.get_clean_model().to(args.device)
    model.eval()
    transform = build_eval_transform(dataset_meta)

    evaluate_original = (not args.skip_original_eval) and all((sample_dir / 'original.png').is_file() for sample_dir in sample_dirs)
    if not args.skip_original_eval and not evaluate_original:
        print('Original-image evaluation skipped because at least one sample is missing original.png')

    compute_quality = evaluate_original and (not args.skip_quality_eval)
    if evaluate_original and args.skip_quality_eval:
        print('Quality-metric evaluation skipped because --skip_quality_eval was provided')

    results = []
    for start in range(0, len(sample_dirs), args.batch_size):
        batch_sample_dirs = sample_dirs[start:start + args.batch_size]
        batch_metadata = metadata_list[start:start + args.batch_size]
        protected_batch = torch.stack([
            load_image_tensor(sample_dir / 'protected.png', transform) for sample_dir in batch_sample_dirs
        ], dim=0).to(args.device)
        protected_metrics = evaluate_tensor_batch(model, protected_batch, target_class, num_classes)

        if evaluate_original:
            original_batch = torch.stack([
                load_image_tensor(sample_dir / 'original.png', transform) for sample_dir in batch_sample_dirs
            ], dim=0).to(args.device)
            original_metrics = evaluate_tensor_batch(model, original_batch, target_class, num_classes)
        else:
            original_metrics = None

        if compute_quality:
            original_raw_batch = torch.stack([
                load_raw_image_tensor(sample_dir / 'original.png') for sample_dir in batch_sample_dirs
            ], dim=0)
            protected_raw_batch = torch.stack([
                load_raw_image_tensor(sample_dir / 'protected.png') for sample_dir in batch_sample_dirs
            ], dim=0)
            quality_metrics = compute_batch_quality_metrics(original_raw_batch, protected_raw_batch)
        else:
            quality_metrics = None

        for idx, (sample_dir, metadata) in enumerate(zip(batch_sample_dirs, batch_metadata)):
            source_class_index = metadata.get('source_class_index')
            if source_class_index is not None:
                source_class_index = int(source_class_index)
            source_class_text = metadata.get('source_text')
            if source_class_text is None and source_class_index is not None:
                source_class_text = label_dict.get(source_class_index)

            result = {
                'sample_dir': str(sample_dir),
                'image_id': metadata.get('image_id', sample_dir.name),
                'source_class_index': source_class_index,
                'source_class_text': source_class_text,
                'target_class_index': target_class,
                'target_class_text': label_dict[target_class],
                'protected_prediction_index': int(protected_metrics['prediction'][idx].item()),
                'protected_prediction_text': label_dict.get(int(protected_metrics['prediction'][idx].item())),
                'protected_target_top1_hit': bool(protected_metrics['top1_hit'][idx].item()),
                'protected_target_top5_hit': bool(protected_metrics['top5_hit'][idx].item()),
                'protected_target_rank': int(protected_metrics['target_rank'][idx].item()),
                'protected_target_prob': float(protected_metrics['target_prob'][idx].item()),
                'protected_target_logit': float(protected_metrics['target_logit'][idx].item()),
            }

            if original_metrics is not None:
                result.update({
                    'original_prediction_index': int(original_metrics['prediction'][idx].item()),
                    'original_prediction_text': label_dict.get(int(original_metrics['prediction'][idx].item())),
                    'original_target_top1_hit': bool(original_metrics['top1_hit'][idx].item()),
                    'original_target_top5_hit': bool(original_metrics['top5_hit'][idx].item()),
                    'original_target_rank': int(original_metrics['target_rank'][idx].item()),
                    'original_target_prob': float(original_metrics['target_prob'][idx].item()),
                    'original_target_logit': float(original_metrics['target_logit'][idx].item()),
                })
                result.update({
                    'target_rank_improvement': result['original_target_rank'] - result['protected_target_rank'],
                    'target_prob_increase': result['protected_target_prob'] - result['original_target_prob'],
                    'target_logit_increase': result['protected_target_logit'] - result['original_target_logit'],
                })

            if quality_metrics is not None:
                for key in QUALITY_KEYS:
                    result[key] = float(quality_metrics[key][idx].item())

            results.append(result)
            if args.verbose:
                message = (
                    f"[{sample_dir.name}] protected_pred={result['protected_prediction_index']} ({result['protected_prediction_text']}), "
                    f"source={format_source_group_label(result['source_class_index'], result['source_class_text'])}, "
                    f"target_rank={result['protected_target_rank']}, target_prob={result['protected_target_prob']:.6f}, "
                    f"top1={result['protected_target_top1_hit']}, top5={result['protected_target_top5_hit']}"
                )
                if quality_metrics is not None:
                    message += (
                        f", L2={result['l2_distance']:.4f}, Linf={result['linf_distance']:.4f}, "
                        f"PSNR={result['psnr']:.2f}, SSIM={result['ssim']:.4f}"
                    )
                print(message)

    source_group_summaries = summarize_source_groups(results, evaluate_original)
    summary = {
        'samples_path': str(Path(args.samples_path)),
        'tgt_dataset': tgt_dataset,
        'target_class_index': target_class,
        'target_class_text': label_dict[target_class],
        'num_classes': num_classes,
        'evaluate_original': evaluate_original,
        'compute_quality_metrics': compute_quality,
        **summarize_records(results, 'protected'),
        'source_group_summaries': source_group_summaries,
        'results': results,
    }

    if evaluate_original:
        summary.update(summarize_records(results, 'original'))
        summary.update({
            'avg_target_rank_improvement': float(sum(record['target_rank_improvement'] for record in results) / len(results)),
            'avg_target_prob_increase': float(sum(record['target_prob_increase'] for record in results) / len(results)),
            'avg_target_logit_increase': float(sum(record['target_logit_increase'] for record in results) / len(results)),
        })

    output_json = Path(args.output_json) if args.output_json else None
    if output_json is None:
        if len(sample_dirs) == 1:
            output_json = sample_dirs[0] / 'evaluation.json'
        else:
            output_json = Path(args.samples_path) / 'evaluation_summary.json'

    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w') as handle:
        json.dump(summary, handle, indent=2)

    print(f'Saved evaluation summary to {output_json}')
    print(
        f"Protected target Top-1 rate: {summary['protected_target_top1_rate']:.4f} "
        f"({summary['protected_target_top1_hits']}/{summary['protected_num_samples']})"
    )
    print(
        f"Protected target Top-5 rate: {summary['protected_target_top5_rate']:.4f} "
        f"({summary['protected_target_top5_hits']}/{summary['protected_num_samples']})"
    )
    print(
        f"Protected avg target rank: {summary['avg_protected_target_rank']:.2f}, "
        f"avg target prob: {summary['avg_protected_target_prob']:.6f}, "
        f"avg target logit: {summary['avg_protected_target_logit']:.6f}"
    )

    if evaluate_original:
        print(
            f"Original target Top-1 rate: {summary['original_target_top1_rate']:.4f}, "
            f"Top-5 rate: {summary['original_target_top5_rate']:.4f}"
        )
        print(
            f"Average target-rank improvement: {summary['avg_target_rank_improvement']:.2f}, "
            f"target-prob increase: {summary['avg_target_prob_increase']:.6f}, "
            f"target-logit increase: {summary['avg_target_logit_increase']:.6f}"
        )

    if compute_quality:
        print(
            f"Average quality metrics: L2={summary['avg_l2_distance']:.4f}, "
            f"Linf={summary['avg_linf_distance']:.4f}, PSNR={summary['avg_psnr']:.2f}, SSIM={summary['avg_ssim']:.4f}"
        )

    print_source_group_summaries(source_group_summaries, evaluate_original)


if __name__ == '__main__':
    main()

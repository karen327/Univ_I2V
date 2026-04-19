import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps
from torchvision import transforms

from utils.quality_metrics import compute_batch_quality_metrics, summarize_quality_records

QUALITY_KEYS = ['l2_distance', 'linf_distance', 'psnr', 'ssim']
RAW_IMAGE_TRANSFORM = transforms.ToTensor()
FONT_REGULAR = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
FONT_BOLD = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'


def parse_args():
    parser = argparse.ArgumentParser(description='Generate a Stage A calibration report for two per-image candidate runs.')
    parser.add_argument('--candidate_a', required=True, help='First candidate run directory.')
    parser.add_argument('--candidate_b', required=True, help='Second candidate run directory.')
    parser.add_argument('--output_dir', default='', help='Directory used to save the Stage A report artifacts.')
    parser.add_argument('--selection_metric', choices=['top5', 'rank_improvement'], default='top5', help='Metric used to decide easy/hard source classes.')
    parser.add_argument('--num_easy_classes', type=int, default=2, help='Number of easy source classes to visualize.')
    parser.add_argument('--num_hard_classes', type=int, default=2, help='Number of hard source classes to visualize.')
    parser.add_argument('--samples_per_class', type=int, default=1, help='Representative samples selected per class for the qualitative wall.')
    parser.add_argument('--quality_batch_size', type=int, default=32, help='Batch size used when quality metrics need to be computed from saved images.')
    return parser.parse_args()


def load_font(size, bold=False):
    font_path = FONT_BOLD if bold else FONT_REGULAR
    try:
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def load_summary(candidate_dir):
    candidate_dir = Path(candidate_dir)
    summary_path = candidate_dir / 'evaluation_summary.json'
    if not summary_path.is_file():
        raise FileNotFoundError(f'Missing evaluation_summary.json: {summary_path}')
    with open(summary_path, 'r') as handle:
        summary = json.load(handle)
    return candidate_dir, summary


def load_raw_image_tensor(image_path):
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        return RAW_IMAGE_TRANSFORM(image)


def chunk_items(items, chunk_size):
    chunk = []
    for item in items:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def compute_quality_for_candidate(candidate_dir, batch_size=32):
    sample_dirs = sorted([path for path in Path(candidate_dir).iterdir() if path.is_dir() and (path / 'protected.png').is_file()])
    if not sample_dirs:
        raise RuntimeError(f'No protected sample directories found under {candidate_dir}')

    per_sample = {}
    records = []
    for batch_sample_dirs in chunk_items(sample_dirs, batch_size):
        original_batch = torch.stack([load_raw_image_tensor(sample_dir / 'original.png') for sample_dir in batch_sample_dirs], dim=0)
        protected_batch = torch.stack([load_raw_image_tensor(sample_dir / 'protected.png') for sample_dir in batch_sample_dirs], dim=0)
        batch_metrics = compute_batch_quality_metrics(original_batch, protected_batch)
        for index, sample_dir in enumerate(batch_sample_dirs):
            metrics = {key: float(batch_metrics[key][index].item()) for key in QUALITY_KEYS}
            per_sample[str(sample_dir)] = metrics
            records.append(metrics)

    return summarize_quality_records(records), per_sample


def enrich_summary_with_quality(summary, quality_by_sample):
    result_records = summary['results']
    for record in result_records:
        sample_key = str(Path(record['sample_dir']))
        metrics = quality_by_sample.get(sample_key)
        if metrics is None:
            continue
        record.update(metrics)

    summary.update(summarize_quality_records([{key: record[key] for key in QUALITY_KEYS} for record in result_records if all(key in record for key in QUALITY_KEYS)]))
    for group_summary in summary.get('source_group_summaries', []):
        group_key = group_summary['source_group_key']
        group_records = [
            record for record in result_records
            if build_source_group_key(record.get('source_class_index'), record.get('source_class_text')) == group_key
            and all(key in record for key in QUALITY_KEYS)
        ]
        group_summary.update(summarize_quality_records([{key: record[key] for key in QUALITY_KEYS} for record in group_records]))


def build_source_group_key(source_class_index, source_class_text):
    if source_class_index is not None:
        return f'{int(source_class_index):04d}:{source_class_text}'
    if source_class_text:
        return str(source_class_text)
    return 'unknown'


def ensure_quality_metrics(summary, candidate_dir, batch_size):
    if all(key in summary for key in ['avg_l2_distance', 'avg_linf_distance', 'avg_psnr', 'avg_ssim']):
        return summary
    aggregate_quality, per_sample_quality = compute_quality_for_candidate(candidate_dir, batch_size=batch_size)
    summary.update(aggregate_quality)
    enrich_summary_with_quality(summary, per_sample_quality)
    return summary


def parse_candidate_id(candidate_dir):
    name = Path(candidate_dir).name
    tokens = []
    for token in name.split('_'):
        if token in {'dataset', 'ImageNet', 'target', 'img', 'eps'}:
            continue
    parts = name.split('_')
    selected = []
    for idx, token in enumerate(parts):
        if token in {'aug', 'tv', 'par'} and idx + 1 < len(parts):
            selected.extend([token, parts[idx + 1]])
    if selected:
        return '_'.join(selected)
    return name


def to_candidate_row(candidate_id, summary):
    return {
        'candidate_id': candidate_id,
        'top1': summary['protected_target_top1_rate'],
        'top5': summary['protected_target_top5_rate'],
        'avg_rank': summary['avg_protected_target_rank'],
        'rank_improvement': summary['avg_target_rank_improvement'],
        'avg_logit_increase': summary['avg_target_logit_increase'],
        'avg_prob': summary['avg_protected_target_prob'],
        'l2_distance': summary['avg_l2_distance'],
        'linf_distance': summary['avg_linf_distance'],
        'psnr': summary['avg_psnr'],
        'ssim': summary['avg_ssim'],
    }


def recommend_primary_candidate(row_a, row_b):
    quality_close = (
        abs(row_a['psnr'] - row_b['psnr']) < 0.5
        and abs(row_a['ssim'] - row_b['ssim']) < 0.01
        and abs(row_a['l2_distance'] - row_b['l2_distance']) / max(min(row_a['l2_distance'], row_b['l2_distance']), 1e-6) < 0.05
    )
    if quality_close:
        if row_a['top5'] >= row_b['top5']:
            return row_a['candidate_id'], row_b['candidate_id'], '两者质量差异较小，按默认优先级保留 proxy 更强的候选作为主候选。'
        return row_b['candidate_id'], row_a['candidate_id'], '两者质量差异较小，按默认优先级保留 proxy 更强的候选作为主候选。'

    a_quality_better = row_a['psnr'] > row_b['psnr'] + 0.5 or row_a['ssim'] > row_b['ssim'] + 0.01 or row_a['l2_distance'] < row_b['l2_distance'] * 0.95
    b_quality_better = row_b['psnr'] > row_a['psnr'] + 0.5 or row_b['ssim'] > row_a['ssim'] + 0.01 or row_b['l2_distance'] < row_a['l2_distance'] * 0.95

    if a_quality_better and row_b['top5'] <= row_a['top5'] + 0.01 and row_b['top1'] <= row_a['top1'] + 0.01:
        return row_a['candidate_id'], row_b['candidate_id'], '候选 A 的视觉质量优势明显，同时 proxy 没有被候选 B 明显超出。'
    if b_quality_better and row_a['top5'] <= row_b['top5'] + 0.01 and row_a['top1'] <= row_b['top1'] + 0.01:
        return row_b['candidate_id'], row_a['candidate_id'], '候选 B 的视觉质量优势明显，同时 proxy 没有被候选 A 明显超出。'

    if row_a['top5'] >= row_b['top5']:
        return row_a['candidate_id'], row_b['candidate_id'], '质量差异不足以压过 proxy 指标差异，优先保留 Top-5 更高的候选。'
    return row_b['candidate_id'], row_a['candidate_id'], '质量差异不足以压过 proxy 指标差异，优先保留 Top-5 更高的候选。'


def build_group_lookup(summary):
    return {group['source_group_key']: group for group in summary['source_group_summaries']}


def select_easy_hard_classes(summary_a, summary_b, selection_metric, num_easy_classes, num_hard_classes):
    group_lookup_a = build_group_lookup(summary_a)
    group_lookup_b = build_group_lookup(summary_b)
    common_keys = sorted(set(group_lookup_a.keys()) & set(group_lookup_b.keys()))
    metric_key = 'protected_target_top5_rate' if selection_metric == 'top5' else 'avg_target_rank_improvement'
    scores = []
    for group_key in common_keys:
        group_a = group_lookup_a[group_key]
        group_b = group_lookup_b[group_key]
        mean_score = (group_a[metric_key] + group_b[metric_key]) / 2.0
        scores.append({
            'source_group_key': group_key,
            'source_class_index': group_a.get('source_class_index'),
            'source_class_text': group_a.get('source_class_text'),
            'mean_score': mean_score,
            'candidate_a_score': group_a[metric_key],
            'candidate_b_score': group_b[metric_key],
        })
    scores.sort(key=lambda item: item['mean_score'], reverse=True)
    easy = scores[:num_easy_classes]
    hard = scores[-num_hard_classes:] if num_hard_classes > 0 else []
    return easy, hard


def build_result_lookup(summary):
    return {str(result['image_id']): result for result in summary['results']}


def build_sample_dir_lookup(candidate_dir):
    lookup = {}
    for sample_dir in Path(candidate_dir).iterdir():
        if not sample_dir.is_dir() or not (sample_dir / 'log.json').is_file():
            continue
        with open(sample_dir / 'log.json', 'r') as handle:
            metadata = json.load(handle)
        lookup[str(metadata.get('image_id'))] = sample_dir
    return lookup


def select_representative_sample_ids(summary_a, summary_b, class_keys, samples_per_class):
    results_a = summary_a['results']
    results_b = summary_b['results']
    lookup_a = build_result_lookup(summary_a)
    lookup_b = build_result_lookup(summary_b)
    selections = []
    for class_info in class_keys:
        group_key = class_info['source_group_key']
        class_results_a = [record for record in results_a if build_source_group_key(record.get('source_class_index'), record.get('source_class_text')) == group_key]
        class_results_b = [record for record in results_b if build_source_group_key(record.get('source_class_index'), record.get('source_class_text')) == group_key]
        shared_ids = sorted(set(str(record['image_id']) for record in class_results_a) & set(str(record['image_id']) for record in class_results_b))
        ranked = []
        for image_id in shared_ids:
            result_a = lookup_a[image_id]
            result_b = lookup_b[image_id]
            mean_rank_improvement = (result_a['target_rank_improvement'] + result_b['target_rank_improvement']) / 2.0
            ranked.append((mean_rank_improvement, image_id))
        ranked.sort(key=lambda item: item[0])
        if not ranked:
            continue
        if samples_per_class == 1:
            chosen_positions = [len(ranked) // 2]
        else:
            chosen_positions = []
            for idx in range(samples_per_class):
                fraction = (idx + 1) / (samples_per_class + 1)
                position = min(len(ranked) - 1, max(0, round(fraction * (len(ranked) - 1))))
                chosen_positions.append(position)
        chosen_positions = sorted(set(chosen_positions))
        selections.append({
            'class_info': class_info,
            'image_ids': [ranked[position][1] for position in chosen_positions],
        })
    return selections


def load_preview_image(image_path, target_size):
    with Image.open(image_path) as image:
        image = image.convert('RGB')
        return ImageOps.fit(image, target_size, method=Image.Resampling.BICUBIC)


def draw_qualitative_wall(selected_samples, candidate_dirs, sample_dir_lookups, output_path):
    candidate_ids = list(candidate_dirs.keys())
    sample_entries = []
    for bucket_name, bucket in selected_samples.items():
        for class_entry in bucket:
            class_label = f"{class_entry['class_info']['source_class_text']} ({bucket_name})"
            for image_id in class_entry['image_ids']:
                sample_entries.append((class_label, image_id))

    cell_size = (180, 180)
    left_width = 260
    gap = 12
    headers = ['Original']
    for candidate_id in candidate_ids:
        headers.extend([f'{candidate_id}\nprotected', f'{candidate_id}\ndelta', f'{candidate_id}\ncompare'])
    cols = len(headers)
    width = left_width + cols * (cell_size[0] + gap) + 40
    row_height = cell_size[1] + 42
    height = 120 + len(sample_entries) * row_height + 40

    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    title_font = load_font(28, bold=True)
    header_font = load_font(18, bold=True)
    text_font = load_font(16)
    draw.text((24, 20), 'Stage A Qualitative Comparison Wall', fill='black', font=title_font)
    draw.text((24, 56), 'Rows cover easy and hard source classes; both candidates use the same representative images.', fill='#444444', font=text_font)

    x = left_width
    for index, header in enumerate(headers):
        cell_x = x + index * (cell_size[0] + gap)
        draw.rounded_rectangle((cell_x, 82, cell_x + cell_size[0], 112), radius=8, fill='#eef3fb', outline='#d2dbe9')
        for line_index, line in enumerate(header.split('\n')):
            bbox = draw.textbbox((0, 0), line, font=header_font)
            draw.text((cell_x + (cell_size[0] - (bbox[2] - bbox[0])) / 2, 88 + line_index * 14), line, fill='black', font=header_font)

    reference_lookup = next(iter(sample_dir_lookups.values()))
    for row_index, (class_label, image_id) in enumerate(sample_entries):
        image_key = str(image_id)
        y = 126 + row_index * row_height
        draw.rounded_rectangle((20, y, left_width - 12, y + cell_size[1]), radius=10, fill='#f7f9fc', outline='#d8e1ee')
        draw.text((34, y + 22), class_label, fill='black', font=text_font)
        draw.text((34, y + 54), f'image_id={image_id}', fill='#555555', font=text_font)

        original_sample_dir = reference_lookup.get(image_key)
        if original_sample_dir is None:
            continue

        cell_images = [load_preview_image(original_sample_dir / 'original.png', cell_size)]
        for candidate_id in candidate_ids:
            sample_dir = sample_dir_lookups[candidate_id].get(image_key)
            if sample_dir is None:
                cell_images.extend([Image.new('RGB', cell_size, '#f0f0f0')] * 3)
                continue
            cell_images.extend([
                load_preview_image(sample_dir / 'protected.png', cell_size),
                load_preview_image(sample_dir / 'delta_vis.png', cell_size),
                load_preview_image(sample_dir / 'compare.png', cell_size),
            ])

        for cell_index, preview in enumerate(cell_images):
            cell_x = left_width + cell_index * (cell_size[0] + gap)
            image.paste(preview, (cell_x, y))
            draw.rounded_rectangle((cell_x, y, cell_x + cell_size[0], y + cell_size[1]), radius=8, outline='#d2dbe9', width=1)

    image.save(output_path)


def write_csv(path, headers, rows):
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(row[header]) for header in headers) + ' |')
    return '\n'.join(lines)


def main():
    args = parse_args()
    candidate_dir_a, summary_a = load_summary(args.candidate_a)
    candidate_dir_b, summary_b = load_summary(args.candidate_b)

    summary_a = ensure_quality_metrics(summary_a, candidate_dir_a, args.quality_batch_size)
    summary_b = ensure_quality_metrics(summary_b, candidate_dir_b, args.quality_batch_size)

    candidate_id_a = parse_candidate_id(candidate_dir_a)
    candidate_id_b = parse_candidate_id(candidate_dir_b)
    row_a = to_candidate_row(candidate_id_a, summary_a)
    row_b = to_candidate_row(candidate_id_b, summary_b)
    primary_candidate_id, backup_candidate_id, recommendation_reason = recommend_primary_candidate(row_a, row_b)

    easy_classes, hard_classes = select_easy_hard_classes(
        summary_a,
        summary_b,
        args.selection_metric,
        args.num_easy_classes,
        args.num_hard_classes,
    )
    selected_easy = select_representative_sample_ids(summary_a, summary_b, easy_classes, args.samples_per_class)
    selected_hard = select_representative_sample_ids(summary_a, summary_b, hard_classes, args.samples_per_class)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(args.output_dir) if args.output_dir else Path('experiments/stage_a_report') / f'{candidate_id_a}_vs_{candidate_id_b}_{timestamp}'
    output_dir.mkdir(parents=True, exist_ok=True)

    aggregate_rows = [row_a, row_b]
    aggregate_headers = ['candidate_id', 'top1', 'top5', 'avg_rank', 'rank_improvement', 'avg_logit_increase', 'avg_prob', 'l2_distance', 'linf_distance', 'psnr', 'ssim']
    write_csv(output_dir / 'aggregate_metrics.csv', aggregate_headers, aggregate_rows)

    group_lookup_a = build_group_lookup(summary_a)
    group_lookup_b = build_group_lookup(summary_b)
    source_group_rows = []
    for group_key in sorted(set(group_lookup_a.keys()) & set(group_lookup_b.keys())):
        group_a = group_lookup_a[group_key]
        group_b = group_lookup_b[group_key]
        source_group_rows.append({
            'source_group_key': group_key,
            'source_class_text': group_a.get('source_class_text'),
            f'{candidate_id_a}_top5': f"{group_a['protected_target_top5_rate']:.4f}",
            f'{candidate_id_b}_top5': f"{group_b['protected_target_top5_rate']:.4f}",
            f'{candidate_id_a}_rank_improvement': f"{group_a['avg_target_rank_improvement']:.2f}",
            f'{candidate_id_b}_rank_improvement': f"{group_b['avg_target_rank_improvement']:.2f}",
            f'{candidate_id_a}_psnr': f"{group_a.get('avg_psnr', float('nan')):.2f}",
            f'{candidate_id_b}_psnr': f"{group_b.get('avg_psnr', float('nan')):.2f}",
            f'{candidate_id_a}_ssim': f"{group_a.get('avg_ssim', float('nan')):.4f}",
            f'{candidate_id_b}_ssim': f"{group_b.get('avg_ssim', float('nan')):.4f}",
        })
    source_group_headers = list(source_group_rows[0].keys()) if source_group_rows else ['source_group_key']
    write_csv(output_dir / 'source_group_comparison.csv', source_group_headers, source_group_rows)

    selected_samples = {'easy': selected_easy, 'hard': selected_hard}
    with open(output_dir / 'selected_samples.json', 'w', encoding='utf-8') as handle:
        json.dump(selected_samples, handle, indent=2, ensure_ascii=False)

    candidate_dirs = {candidate_id_a: candidate_dir_a, candidate_id_b: candidate_dir_b}
    result_lookups = {candidate_id_a: build_result_lookup(summary_a), candidate_id_b: build_result_lookup(summary_b)}
    sample_dir_lookups = {candidate_id_a: build_sample_dir_lookup(candidate_dir_a), candidate_id_b: build_sample_dir_lookup(candidate_dir_b)}

    artifact_rows = []
    for bucket_name, bucket_entries in selected_samples.items():
        for class_entry in bucket_entries:
            source_class_text = class_entry['class_info']['source_class_text']
            for image_id in class_entry['image_ids']:
                image_key = str(image_id)
                for candidate_id in [candidate_id_a, candidate_id_b]:
                    result_record = result_lookups[candidate_id].get(image_key, {})
                    sample_dir = sample_dir_lookups[candidate_id].get(image_key)
                    original_dir = sample_dir_lookups[candidate_id_a].get(image_key)
                    artifact_rows.append({
                        'candidate_id': candidate_id,
                        'source_bucket': bucket_name,
                        'source_class_text': source_class_text,
                        'image_id': image_id,
                        'original_image_path': str((original_dir / 'original.png').resolve()) if original_dir else '',
                        'protected_image_path': str((sample_dir / 'protected.png').resolve()) if sample_dir else '',
                        'delta_vis_path': str((sample_dir / 'delta_vis.png').resolve()) if sample_dir else '',
                        'compare_image_path': str((sample_dir / 'compare.png').resolve()) if sample_dir else '',
                        'protected_target_rank': result_record.get('protected_target_rank', ''),
                        'target_rank_improvement': result_record.get('target_rank_improvement', ''),
                        'psnr': f"{result_record.get('psnr', float('nan')):.2f}" if 'psnr' in result_record else '',
                        'ssim': f"{result_record.get('ssim', float('nan')):.4f}" if 'ssim' in result_record else '',
                        'artifact_type': '',
                        'artifact_severity': '',
                        'semantic_hallucination': '',
                        'texture_noise': '',
                        'color_shift': '',
                        'structure_distortion': '',
                        'notes': '',
                    })
    artifact_headers = list(artifact_rows[0].keys()) if artifact_rows else ['candidate_id']
    write_csv(output_dir / 'artifact_review_template.csv', artifact_headers, artifact_rows)

    draw_qualitative_wall(selected_samples, candidate_dirs, sample_dir_lookups, output_dir / 'qualitative_comparison_wall.png')

    artifact_guide = [
        '# Artifact-Type Review Guide',
        '',
        '- artifact_type: choose the dominant artifact family, such as `semantic_residue`, `texture_noise`, `color_shift`, `structure_distortion`, or `other`.',
        '- artifact_severity: use a short scale such as `mild`, `moderate`, or `severe`.',
        '- semantic_hallucination / texture_noise / color_shift / structure_distortion: mark `yes` or `no` for quick filtering.',
        '- notes: add one or two short observations about why the protected image looks problematic or acceptable.',
    ]
    with open(output_dir / 'artifact_type_guide.md', 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(artifact_guide) + '\n')

    stage_a_summary = {
        'candidate_a': {'candidate_id': candidate_id_a, 'run_dir': str(candidate_dir_a), **row_a},
        'candidate_b': {'candidate_id': candidate_id_b, 'run_dir': str(candidate_dir_b), **row_b},
        'primary_candidate_id': primary_candidate_id,
        'backup_candidate_id': backup_candidate_id,
        'recommendation_reason': recommendation_reason,
        'selection_metric': args.selection_metric,
        'easy_classes': easy_classes,
        'hard_classes': hard_classes,
        'selected_samples_file': str((output_dir / 'selected_samples.json').resolve()),
        'artifact_review_template': str((output_dir / 'artifact_review_template.csv').resolve()),
        'artifact_type_guide': str((output_dir / 'artifact_type_guide.md').resolve()),
        'qualitative_wall': str((output_dir / 'qualitative_comparison_wall.png').resolve()),
    }
    with open(output_dir / 'stage_a_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(stage_a_summary, handle, indent=2, ensure_ascii=False)

    md_lines = [
        '# Stage A Calibration Report',
        '',
        f'- Candidate A: `{candidate_id_a}`',
        f'- Candidate B: `{candidate_id_b}`',
        f'- Recommended primary candidate: `{primary_candidate_id}`',
        f'- Recommended backup candidate: `{backup_candidate_id}`',
        f'- Recommendation reason: {recommendation_reason}',
        '',
        '## Aggregate Metrics',
        '',
        build_markdown_table(aggregate_headers, [{key: f"{value:.4f}" if isinstance(value, float) else value for key, value in row.items()} for row in aggregate_rows]),
        '',
        '## Source-Class Comparison',
        '',
        build_markdown_table(source_group_headers, source_group_rows),
        '',
        '## Selected Easy/Hard Classes',
        '',
        f'- Easy classes: {", ".join(item["source_class_text"] for item in easy_classes)}',
        f'- Hard classes: {", ".join(item["source_class_text"] for item in hard_classes)}',
        '',
        '## Output Artifacts',
        '',
        f'- Aggregate CSV: `{(output_dir / "aggregate_metrics.csv").name}`',
        f'- Source-group CSV: `{(output_dir / "source_group_comparison.csv").name}`',
        f'- Selected samples JSON: `{(output_dir / "selected_samples.json").name}`',
        f'- Qualitative wall: `{(output_dir / "qualitative_comparison_wall.png").name}`',
        '',
        '## Artifact-Type Review',
        '',
        '- Use the generated CSV template together with the guide below to record qualitative artifact types.',
        f'- Artifact review template: `{(output_dir / "artifact_review_template.csv").name}`',
        f'- Artifact type guide: `{(output_dir / "artifact_type_guide.md").name}`',
    ]
    with open(output_dir / 'stage_a_report.md', 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(md_lines) + '\n')

    print(f'Stage A report saved to {output_dir}')
    print(f'Recommended primary candidate: {primary_candidate_id}')
    print(f'Recommended backup candidate: {backup_candidate_id}')
    print(f'Qualitative wall: {output_dir / "qualitative_comparison_wall.png"}')


if __name__ == '__main__':
    main()

import argparse
import csv
import json
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description='Prepare a lightweight Stage B i2v pilot protocol workspace.')
    parser.add_argument('--primary_candidate_dir', default='', help='Primary candidate run directory. Optional when --stage_a_summary is provided.')
    parser.add_argument('--secondary_candidate_dir', default='', help='Backup candidate run directory. Optional when --stage_a_summary is provided.')
    parser.add_argument('--stage_a_summary', default='', help='Optional stage_a_summary.json used to auto-resolve the primary and backup candidate run directories.')
    parser.add_argument('--output_dir', required=True, help='Output directory used for the Stage B pilot workspace.')
    parser.add_argument('--model_ids', required=True, help='Comma-separated list of 1-2 fixed i2v model identifiers used in this pilot.')
    parser.add_argument('--prompt', required=True, help='Fixed prompt string used by every clean/protected i2v generation pair.')
    parser.add_argument('--seed', type=int, default=0, help='Fixed seed used for the pilot.')
    parser.add_argument('--num_inference_steps', type=int, default=25, help='Fixed i2v inference steps used for the pilot.')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Fixed guidance scale used for the pilot.')
    parser.add_argument('--num_frames', type=int, default=16, help='Fixed number of output frames used for the pilot.')
    parser.add_argument('--frame_width', type=int, default=512, help='Fixed frame width used for the pilot.')
    parser.add_argument('--frame_height', type=int, default=512, help='Fixed frame height used for the pilot.')
    parser.add_argument('--selection_metric', choices=['top5', 'rank_improvement'], default='top5', help='Metric used to select easy and hard source classes.')
    parser.add_argument('--num_easy_classes', type=int, default=2, help='Number of easy source classes to include in the pilot subset.')
    parser.add_argument('--num_hard_classes', type=int, default=2, help='Number of hard source classes to include in the pilot subset.')
    parser.add_argument('--samples_per_class', type=int, default=5, help='Number of representative samples selected per class.')
    parser.add_argument('--skip_asset_copy', action='store_true', help='Do not copy clean/protected still images into the pilot workspace.')
    return parser.parse_args()


def load_json(path):
    with open(path, 'r', encoding='utf-8') as handle:
        return json.load(handle)


def parse_candidate_id(candidate_dir):
    name = Path(candidate_dir).name
    parts = name.split('_')
    selected = []
    for idx, token in enumerate(parts):
        if token in {'aug', 'tv', 'par'} and idx + 1 < len(parts):
            selected.extend([token, parts[idx + 1]])
    return '_'.join(selected) if selected else name


def resolve_candidates(args):
    if args.stage_a_summary:
        stage_a_summary = load_json(args.stage_a_summary)
        primary_dir = stage_a_summary['candidate_a']['run_dir'] if stage_a_summary['candidate_a']['candidate_id'] == stage_a_summary['primary_candidate_id'] else stage_a_summary['candidate_b']['run_dir']
        backup_dir = stage_a_summary['candidate_a']['run_dir'] if stage_a_summary['candidate_a']['candidate_id'] == stage_a_summary['backup_candidate_id'] else stage_a_summary['candidate_b']['run_dir']
        return Path(primary_dir), Path(backup_dir)

    if not args.primary_candidate_dir or not args.secondary_candidate_dir:
        raise ValueError('Either provide --stage_a_summary or provide both --primary_candidate_dir and --secondary_candidate_dir.')
    return Path(args.primary_candidate_dir), Path(args.secondary_candidate_dir)


def build_source_group_key(source_class_index, source_class_text):
    if source_class_index is not None:
        return f'{int(source_class_index):04d}:{source_class_text}'
    if source_class_text:
        return str(source_class_text)
    return 'unknown'


def build_group_lookup(summary):
    return {group['source_group_key']: group for group in summary['source_group_summaries']}


def build_result_lookup(summary):
    return {str(result['image_id']): result for result in summary['results']}


def build_sample_dir_lookup(candidate_dir):
    lookup = {}
    for sample_dir in Path(candidate_dir).iterdir():
        if not sample_dir.is_dir() or not (sample_dir / 'log.json').is_file():
            continue
        metadata = load_json(sample_dir / 'log.json')
        lookup[str(metadata.get('image_id'))] = sample_dir
    return lookup


def select_easy_hard_classes(summary_primary, summary_secondary, selection_metric, num_easy_classes, num_hard_classes):
    metric_key = 'protected_target_top5_rate' if selection_metric == 'top5' else 'avg_target_rank_improvement'
    primary_groups = build_group_lookup(summary_primary)
    secondary_groups = build_group_lookup(summary_secondary)
    common_keys = sorted(set(primary_groups.keys()) & set(secondary_groups.keys()))
    rows = []
    for group_key in common_keys:
        group_primary = primary_groups[group_key]
        group_secondary = secondary_groups[group_key]
        rows.append({
            'source_group_key': group_key,
            'source_class_index': group_primary.get('source_class_index'),
            'source_class_text': group_primary.get('source_class_text'),
            'score': (group_primary[metric_key] + group_secondary[metric_key]) / 2.0,
        })
    rows.sort(key=lambda item: item['score'], reverse=True)
    easy = rows[:num_easy_classes]
    hard = rows[-num_hard_classes:] if num_hard_classes > 0 else []
    return easy, hard


def select_samples(summary_primary, summary_secondary, class_rows, samples_per_class):
    primary_lookup = build_result_lookup(summary_primary)
    secondary_lookup = build_result_lookup(summary_secondary)
    selections = []
    for class_row in class_rows:
        group_key = class_row['source_group_key']
        primary_class_results = [record for record in summary_primary['results'] if build_source_group_key(record.get('source_class_index'), record.get('source_class_text')) == group_key]
        secondary_class_results = [record for record in summary_secondary['results'] if build_source_group_key(record.get('source_class_index'), record.get('source_class_text')) == group_key]
        shared_ids = sorted(set(str(record['image_id']) for record in primary_class_results) & set(str(record['image_id']) for record in secondary_class_results))
        ranked = []
        for image_id in shared_ids:
            result_primary = primary_lookup[image_id]
            result_secondary = secondary_lookup[image_id]
            mean_rank_improvement = (result_primary['target_rank_improvement'] + result_secondary['target_rank_improvement']) / 2.0
            ranked.append((mean_rank_improvement, image_id))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if not ranked:
            continue
        positions = []
        if samples_per_class == 1:
            positions = [len(ranked) // 2]
        else:
            for idx in range(samples_per_class):
                fraction = (idx + 1) / (samples_per_class + 1)
                positions.append(min(len(ranked) - 1, max(0, round(fraction * (len(ranked) - 1)))))
        positions = sorted(set(positions))
        selections.append({
            'class_info': class_row,
            'image_ids': [ranked[position][1] for position in positions],
        })
    return selections


def copy_if_needed(src, dst, skip_copy):
    if skip_copy:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main():
    args = parse_args()
    model_ids = [item.strip() for item in args.model_ids.split(',') if item.strip()]
    if not model_ids:
        raise ValueError('At least one model id must be provided via --model_ids.')
    if len(model_ids) > 2:
        raise ValueError('This lightweight pilot skeleton supports at most 2 fixed i2v model ids.')

    primary_candidate_dir, secondary_candidate_dir = resolve_candidates(args)
    summary_primary = load_json(primary_candidate_dir / 'evaluation_summary.json')
    summary_secondary = load_json(secondary_candidate_dir / 'evaluation_summary.json')
    primary_candidate_id = parse_candidate_id(primary_candidate_dir)
    secondary_candidate_id = parse_candidate_id(secondary_candidate_dir)

    easy_classes, hard_classes = select_easy_hard_classes(
        summary_primary,
        summary_secondary,
        args.selection_metric,
        args.num_easy_classes,
        args.num_hard_classes,
    )
    selected_easy = select_samples(summary_primary, summary_secondary, easy_classes, args.samples_per_class)
    selected_hard = select_samples(summary_primary, summary_secondary, hard_classes, args.samples_per_class)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_dir_lookup_primary = build_sample_dir_lookup(primary_candidate_dir)
    sample_dir_lookup_secondary = build_sample_dir_lookup(secondary_candidate_dir)

    manifest = {
        'protocol': {
            'primary_candidate_id': primary_candidate_id,
            'secondary_candidate_id': secondary_candidate_id,
            'candidate_config_ids': [primary_candidate_id, secondary_candidate_id],
            'i2v_model_ids': model_ids,
            'prompt': args.prompt,
            'seed': args.seed,
            'num_inference_steps': args.num_inference_steps,
            'guidance_scale': args.guidance_scale,
            'num_frames': args.num_frames,
            'frame_resolution': {'width': args.frame_width, 'height': args.frame_height},
            'selection_metric': args.selection_metric,
        },
        'samples': [],
    }

    annotation_rows = []
    grouped_selection = [('easy', selected_easy), ('hard', selected_hard)]
    for bucket_name, bucket_entries in grouped_selection:
        for class_entry in bucket_entries:
            class_info = class_entry['class_info']
            for index, image_id in enumerate(class_entry['image_ids']):
                image_key = str(image_id)
                primary_sample_dir = sample_dir_lookup_primary[image_key]
                secondary_sample_dir = sample_dir_lookup_secondary[image_key]
                sample_tag = f"{bucket_name}_{class_info['source_class_index']}_{index:02d}_{image_key}"
                sample_workspace = output_dir / 'samples' / sample_tag
                clean_image_dst = sample_workspace / 'clean.png'
                primary_image_dst = sample_workspace / primary_candidate_id / 'protected.png'
                secondary_image_dst = sample_workspace / secondary_candidate_id / 'protected.png'
                copy_if_needed(primary_sample_dir / 'original.png', clean_image_dst, args.skip_asset_copy)
                copy_if_needed(primary_sample_dir / 'protected.png', primary_image_dst, args.skip_asset_copy)
                copy_if_needed(primary_sample_dir / 'delta_vis.png', sample_workspace / primary_candidate_id / 'delta_vis.png', args.skip_asset_copy)
                copy_if_needed(primary_sample_dir / 'compare.png', sample_workspace / primary_candidate_id / 'compare.png', args.skip_asset_copy)
                copy_if_needed(secondary_sample_dir / 'protected.png', secondary_image_dst, args.skip_asset_copy)
                copy_if_needed(secondary_sample_dir / 'delta_vis.png', sample_workspace / secondary_candidate_id / 'delta_vis.png', args.skip_asset_copy)
                copy_if_needed(secondary_sample_dir / 'compare.png', sample_workspace / secondary_candidate_id / 'compare.png', args.skip_asset_copy)

                sample_record = {
                    'sample_id': sample_tag,
                    'image_id': image_id,
                    'source_bucket': bucket_name,
                    'source_class_index': class_info['source_class_index'],
                    'source_class_text': class_info['source_class_text'],
                    'clean_image_path': str(clean_image_dst if not args.skip_asset_copy else (primary_sample_dir / 'original.png')),
                    'candidates': {
                        primary_candidate_id: {
                            'protected_image_path': str(primary_image_dst if not args.skip_asset_copy else (primary_sample_dir / 'protected.png')),
                            'delta_vis_path': str((sample_workspace / primary_candidate_id / 'delta_vis.png') if not args.skip_asset_copy else (primary_sample_dir / 'delta_vis.png')),
                            'compare_image_path': str((sample_workspace / primary_candidate_id / 'compare.png') if not args.skip_asset_copy else (primary_sample_dir / 'compare.png')),
                        },
                        secondary_candidate_id: {
                            'protected_image_path': str(secondary_image_dst if not args.skip_asset_copy else (secondary_sample_dir / 'protected.png')),
                            'delta_vis_path': str((sample_workspace / secondary_candidate_id / 'delta_vis.png') if not args.skip_asset_copy else (secondary_sample_dir / 'delta_vis.png')),
                            'compare_image_path': str((sample_workspace / secondary_candidate_id / 'compare.png') if not args.skip_asset_copy else (secondary_sample_dir / 'compare.png')),
                        },
                    },
                    'planned_outputs': {},
                }

                for model_id in model_ids:
                    clean_video_path = output_dir / 'generated' / model_id / sample_tag / 'clean.mp4'
                    sample_record['planned_outputs'][model_id] = {
                        'clean_video_path': str(clean_video_path),
                        primary_candidate_id: str(output_dir / 'generated' / model_id / sample_tag / f'{primary_candidate_id}.mp4'),
                        secondary_candidate_id: str(output_dir / 'generated' / model_id / sample_tag / f'{secondary_candidate_id}.mp4'),
                    }
                    for candidate_id in [primary_candidate_id, secondary_candidate_id]:
                        annotation_rows.append({
                            'sample_id': sample_tag,
                            'image_id': image_id,
                            'source_bucket': bucket_name,
                            'source_class_index': class_info['source_class_index'],
                            'source_class_text': class_info['source_class_text'],
                            'i2v_model_id': model_id,
                            'candidate_config_id': candidate_id,
                            'prompt': args.prompt,
                            'seed': args.seed,
                            'num_inference_steps': args.num_inference_steps,
                            'guidance_scale': args.guidance_scale,
                            'num_frames': args.num_frames,
                            'frame_width': args.frame_width,
                            'frame_height': args.frame_height,
                            'clean_image_path': sample_record['clean_image_path'],
                            'protected_image_path': sample_record['candidates'][candidate_id]['protected_image_path'],
                            'clean_video_path': str(clean_video_path),
                            'protected_video_path': sample_record['planned_outputs'][model_id][candidate_id],
                            'clean_label': '',
                            'protected_label': '',
                            'abnormal_type': '',
                            'notes': '',
                        })
                manifest['samples'].append(sample_record)

    with open(output_dir / 'sample_subset_manifest.json', 'w', encoding='utf-8') as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    protocol_headers = list(annotation_rows[0].keys()) if annotation_rows else ['sample_id']
    with open(output_dir / 'annotation_template.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=protocol_headers)
        writer.writeheader()
        writer.writerows(annotation_rows)

    guide_lines = [
        '# Stage B Pilot Protocol',
        '',
        f'- Primary candidate: `{primary_candidate_id}`',
        f'- Backup candidate: `{secondary_candidate_id}`',
        f'- i2v models: {", ".join(model_ids)}',
        f'- Prompt: `{args.prompt}`',
        f'- Seed: `{args.seed}`',
        f'- Inference steps: `{args.num_inference_steps}`',
        f'- Guidance scale: `{args.guidance_scale}`',
        f'- Frames: `{args.num_frames}`',
        f'- Frame resolution: `{args.frame_width}x{args.frame_height}`',
        '',
        '## Human Annotation Labels',
        '',
        '- clean_label / protected_label: use one of `正常`, `轻微异常`, `明显异常`, `不可用`.',
        '- abnormal_type: short tags such as `semantic_drift`, `flicker`, `collapse`, `motion_artifact`, `other`.',
        '- notes: 1-2 short sentences describing the failure mode.',
        '',
        '## Files',
        '',
        '- sample_subset_manifest.json: fixed protocol and selected pilot subset.',
        '- annotation_template.csv: fill this after generating clean/protected videos.',
    ]
    with open(output_dir / 'pilot_protocol.md', 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(guide_lines) + '\n')

    print(f'Stage B pilot workspace saved to {output_dir}')
    print(f'Selected samples: {len(manifest["samples"])}')
    print(f'Annotation template: {output_dir / "annotation_template.csv"}')


if __name__ == '__main__':
    main()

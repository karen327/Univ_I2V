import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


SEVERE_LABELS = {'明显异常', '不可用', 'severe', 'unusable'}
NORMALIZED_LABEL_MAP = {
    '正常': '正常',
    '轻微异常': '轻微异常',
    '明显异常': '明显异常',
    '不可用': '不可用',
    'normal': '正常',
    'mild': '轻微异常',
    'severe': '明显异常',
    'unusable': '不可用',
}


def parse_args():
    parser = argparse.ArgumentParser(description='Summarize a lightweight Stage B i2v pilot from human annotations.')
    parser.add_argument('--pilot_dir', required=True, help='Stage B pilot workspace directory.')
    parser.add_argument('--annotation_csv', default='', help='Optional annotation CSV path. Defaults to <pilot_dir>/annotation_template.csv.')
    parser.add_argument('--output_dir', default='', help='Directory used to save the pilot summary artifacts.')
    parser.add_argument('--meaningful_signal_threshold', type=float, default=0.30, help='Protected severe-rate threshold used by the default signal gate.')
    parser.add_argument('--max_clean_severe_rate', type=float, default=0.15, help='Maximum clean severe-rate allowed by the default signal gate.')
    return parser.parse_args()


def normalize_label(label):
    label = (label or '').strip()
    return NORMALIZED_LABEL_MAP.get(label, label)


def is_severe(label):
    return normalize_label(label) in SEVERE_LABELS


def load_annotations(annotation_path):
    with open(annotation_path, 'r', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader]


def summarize_rows(rows, meaningful_signal_threshold, max_clean_severe_rate):
    grouped = defaultdict(list)
    for row in rows:
        key = (row['i2v_model_id'], row['candidate_config_id'])
        grouped[key].append(row)

    group_summaries = []
    for (model_id, candidate_id), group_rows in sorted(grouped.items()):
        labeled_rows = [row for row in group_rows if row.get('clean_label') and row.get('protected_label')]
        if labeled_rows:
            clean_severe_count = sum(is_severe(row['clean_label']) for row in labeled_rows)
            protected_severe_count = sum(is_severe(row['protected_label']) for row in labeled_rows)
            protected_labels = defaultdict(int)
            for row in labeled_rows:
                protected_labels[normalize_label(row['protected_label'])] += 1
            clean_severe_rate = clean_severe_count / len(labeled_rows)
            protected_severe_rate = protected_severe_count / len(labeled_rows)
        else:
            clean_severe_count = 0
            protected_severe_count = 0
            protected_labels = defaultdict(int)
            clean_severe_rate = 0.0
            protected_severe_rate = 0.0

        passes_default_gate = (
            len(labeled_rows) > 0
            and protected_severe_rate >= meaningful_signal_threshold
            and clean_severe_rate <= max_clean_severe_rate
            and protected_severe_rate > clean_severe_rate
        )

        group_summaries.append({
            'i2v_model_id': model_id,
            'candidate_config_id': candidate_id,
            'num_rows': len(group_rows),
            'num_labeled_rows': len(labeled_rows),
            'clean_severe_count': clean_severe_count,
            'protected_severe_count': protected_severe_count,
            'clean_severe_rate': round(clean_severe_rate, 4),
            'protected_severe_rate': round(protected_severe_rate, 4),
            'severe_rate_delta': round(protected_severe_rate - clean_severe_rate, 4),
            'protected_normal_count': protected_labels['正常'],
            'protected_mild_count': protected_labels['轻微异常'],
            'protected_obvious_count': protected_labels['明显异常'],
            'protected_unusable_count': protected_labels['不可用'],
            'passes_default_signal_gate': passes_default_gate,
        })
    return group_summaries


def build_candidate_rollup(group_summaries):
    rollup = defaultdict(list)
    for summary in group_summaries:
        rollup[summary['candidate_config_id']].append(summary)

    rows = []
    for candidate_id, candidate_rows in sorted(rollup.items()):
        rows.append({
            'candidate_config_id': candidate_id,
            'num_models': len(candidate_rows),
            'mean_clean_severe_rate': round(sum(row['clean_severe_rate'] for row in candidate_rows) / len(candidate_rows), 4),
            'mean_protected_severe_rate': round(sum(row['protected_severe_rate'] for row in candidate_rows) / len(candidate_rows), 4),
            'mean_severe_rate_delta': round(sum(row['severe_rate_delta'] for row in candidate_rows) / len(candidate_rows), 4),
            'num_models_passing_default_gate': sum(row['passes_default_signal_gate'] for row in candidate_rows),
        })
    return rows


def write_csv(path, rows):
    headers = list(rows[0].keys()) if rows else ['candidate_config_id']
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def build_markdown_table(rows):
    if not rows:
        return '| empty |\n|---|'
    headers = list(rows[0].keys())
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|']
    for row in rows:
        lines.append('| ' + ' | '.join(str(row[header]) for header in headers) + ' |')
    return '\n'.join(lines)


def main():
    args = parse_args()
    pilot_dir = Path(args.pilot_dir)
    annotation_path = Path(args.annotation_csv) if args.annotation_csv else pilot_dir / 'annotation_template.csv'
    if not annotation_path.is_file():
        raise FileNotFoundError(f'Annotation CSV not found: {annotation_path}')

    output_dir = Path(args.output_dir) if args.output_dir else pilot_dir / 'pilot_summary'
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_annotations(annotation_path)
    group_summaries = summarize_rows(rows, args.meaningful_signal_threshold, args.max_clean_severe_rate)
    candidate_rollup = build_candidate_rollup(group_summaries)

    write_csv(output_dir / 'pilot_group_summary.csv', group_summaries)
    write_csv(output_dir / 'pilot_candidate_summary.csv', candidate_rollup)

    summary_json = {
        'annotation_csv': str(annotation_path),
        'meaningful_signal_threshold': args.meaningful_signal_threshold,
        'max_clean_severe_rate': args.max_clean_severe_rate,
        'group_summaries': group_summaries,
        'candidate_rollup': candidate_rollup,
    }
    with open(output_dir / 'pilot_summary.json', 'w', encoding='utf-8') as handle:
        json.dump(summary_json, handle, indent=2, ensure_ascii=False)

    md_lines = [
        '# Stage B Pilot Summary',
        '',
        '## Per-Model / Per-Candidate Summary',
        '',
        build_markdown_table(group_summaries),
        '',
        '## Candidate Rollup',
        '',
        build_markdown_table(candidate_rollup),
        '',
        '## Default Signal Gate',
        '',
        f'- meaningful_signal_threshold: `{args.meaningful_signal_threshold}`',
        f'- max_clean_severe_rate: `{args.max_clean_severe_rate}`',
        '- pass condition: protected severe-rate >= threshold, clean severe-rate <= max_clean_severe_rate, and protected severe-rate > clean severe-rate.',
    ]
    with open(output_dir / 'pilot_summary.md', 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(md_lines) + '\n')

    print(f'Stage B pilot summary saved to {output_dir}')
    for row in group_summaries:
        print(
            f"[{row['i2v_model_id']} | {row['candidate_config_id']}] "
            f"protected_severe_rate={row['protected_severe_rate']:.4f}, "
            f"clean_severe_rate={row['clean_severe_rate']:.4f}, "
            f"passes_default_gate={row['passes_default_signal_gate']}"
        )


if __name__ == '__main__':
    main()

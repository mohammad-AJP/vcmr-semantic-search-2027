#!/usr/bin/env python
"""Build reproducible episode-grouped development splits from original TVR train."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np

from vcmr.data import corpus_videos, episode_id, load_records, normalize_caption, show_id


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')


def split_episodes(records, target_videos, seed):
    videos = corpus_videos(records)
    groups = defaultdict(list)
    for video in videos:
        groups[episode_id(video)].append(video)
    by_show = defaultdict(list)
    for episode, members in groups.items():
        by_show[show_id(members[0])].append(episode)
    rng = np.random.default_rng(seed)
    remaining = set(groups)
    allocations = {'dev_select': set(), 'dev_confirm': set()}
    for split in allocations:
        for show in sorted(by_show):
            available = sorted(e for e in by_show[show] if e in remaining)
            rng.shuffle(available)
            show_total = sum(len(groups[e]) for e in by_show[show])
            target = target_videos * show_total / len(videos)
            chosen, n = [], 0
            for episode in available:
                size = len(groups[episode])
                if abs(n + size - target) < abs(n - target):
                    chosen.append(episode)
                    n += size
            allocations[split].update(chosen)
            remaining.difference_update(chosen)
    allocations['train_fit'] = remaining
    labels = {episode: split for split, episodes in allocations.items() for episode in episodes}
    splits = {split: [] for split in ('train_fit', 'dev_select', 'dev_confirm')}
    for row in records:
        splits[labels[episode_id(row['vid_name'])]].append(row)
    return splits


def make_query_mapping(root, records, source_root):
    # Read source metadata once, then persist identity mapping entirely in new project.
    # Do not infer ordinal feature IDs from official annotation line order.
    by_caption = defaultdict(list)
    for split in ('train', 'val'):
        source = source_root / 'tvr/gprvr' / f'queries_{split}.jsonl'
        for row in load_records(source):
            key = (row['video_id'], normalize_caption(row['caption']))
            by_caption[key].append(row['cap_id'])
    mapping = {}
    duplicate_text_queries = 0
    with h5py.File(root / 'data/tvr/clip_query.h5', 'r') as features:
        for row in records:
            key = (row['vid_name'], normalize_caption(row['desc']))
            candidates = sorted(set(x for x in by_caption[key] if x in features))
            if not candidates:
                raise ValueError(f'No matching CLIP query feature for official desc_id {row["desc_id"]}')
            duplicate_text_queries += len(candidates) > 1
            mapping[str(row['desc_id'])] = candidates[0]
    return mapping, duplicate_text_queries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--source-root', type=Path, default=Path('/home/mohammad/VisualSearch'))
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--dev-videos', type=int, default=1000)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / 'data/tvr'
    train = load_records(base / 'official_train.jsonl')
    val = load_records(base / 'official_val.jsonl')
    splits = split_episodes(train, args.dev_videos, args.seed)
    train_ids = {str(row['desc_id']) for row in train}
    val_ids = {str(row['desc_id']) for row in val}
    assert not train_ids & val_ids, 'Official query identities overlap'
    assert not set(corpus_videos(train)) & set(corpus_videos(val)), 'Official videos overlap'
    split_episode_sets = {name: {episode_id(v) for v in corpus_videos(rows)} for name, rows in splits.items()}
    for a in splits:
        for b in splits:
            if a != b:
                assert not split_episode_sets[a] & split_episode_sets[b]
    assert sum(len(rows) for rows in splits.values()) == len(train)
    stats = {}
    for split, rows in splits.items():
        path = base / 'splits' / f'{split}.jsonl'
        path.parent.mkdir(parents=True, exist_ok=True)
        content = ''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in rows)
        if path.exists() and path.read_text() != content:
            raise RuntimeError(f'Refusing to alter existing frozen split: {path}')
        path.write_text(content)
        videos = corpus_videos(rows)
        stats[split] = {'queries': len(rows), 'videos': len(videos), 'episodes': len(split_episode_sets[split]),
                        'show_video_counts': dict(Counter(show_id(v) for v in videos)),
                        'query_types': dict(Counter(row['type'] for row in rows)), 'sha256': digest(path)}
    mapping, repeated_text = make_query_mapping(root, train + val, args.source_root)
    save_json(root / 'manifests/tvr_clip_query_ids.json', mapping)
    timing_counts = Counter()
    timing_outliers = []
    val_unique = {row['vid_name']: row for row in val}
    with h5py.File(base / 'clip_video.h5', 'r') as features:
        feature_missing = [v for v in corpus_videos(train + val) if v not in features]
        if feature_missing:
            raise ValueError(f'Missing video features: {feature_missing[:5]}')
        for video, row in val_unique.items():
            n = len(features[video])
            difference = n - int(np.ceil(row['duration'] / 1.5))
            timing_counts[difference] += 1
            if abs(difference) >= 2:
                timing_outliers.append({'vid_name': video, 'n_features': n, 'duration': row['duration'], 'difference_from_ceil_dur_over_1p5': difference})
    manifest = {
        'dataset': 'TVR', 'seed': args.seed, 'development_target_videos_each': args.dev_videos,
        'split_strategy': 'source episode groups, deterministic shuffled greedy sampling approximately stratified by TV show',
        'splits': stats,
        'official_val': {'queries': len(val), 'videos': len(val_unique), 'query_types': dict(Counter(row['type'] for row in val)),
                         'sha256': digest(base / 'official_val.jsonl'), 'unchanged_source_annotations': True},
        'official_train_sha256': digest(base / 'official_train.jsonl'),
        'leakage_controls': {
            'no_official_val_queries_for_fitting_or_development': True,
            'episode_disjoint_new_scoring_development': True,
            'official_train_val_episode_overlap_count': len({episode_id(v) for v in corpus_videos(train)} & {episode_id(v) for v in corpus_videos(val)}),
            'official_split_note': 'Official train and validation are video-disjoint but share source episodes. The new episode grouping applies only between train_fit, dev_select, and dev_confirm; the official benchmark split is preserved.',
            'backbone_caveat': 'Fixed DeMUL pretrained baseline was trained on the original full TVR train split, including development videos. Development estimates are optimistic for this fixed backbone; new scoring parameters are selected only on development splits. Official validation remains evaluation-only.',
            'prediction_policy': 'No ground-truth video or moment insertion. Corpus candidates depend on features/query only.',
            'local_test_alias': 'Original local test caption file equals validation; never treat it as an independent test split.'},
        'clip_query_identity': {'mapped_official_query_ids': len(mapping), 'query_rows_with_multiple_identical_text_features': repeated_text,
                               'method': 'Match video ID and normalized caption to original prepared metadata; deterministic feature key among identical captions.',
                               'mapping_file': 'manifests/tvr_clip_query_ids.json'},
        'timing': {'clip_feature_timestamps_present': False,
                   'nominal_TVR_stride_seconds': 1.5,
                   'clip_experiment_fallback': 'Uniform feature cells over [0, annotated duration] unless original feature timing is independently verified.',
                   'demul_policy': 'Use original supplied backbone feature grid/decoder; do not substitute CLIP-derived timestamp assumptions.',
                   'validation_feature_count_minus_ceil_duration_over_1p5': {str(k): v for k, v in sorted(timing_counts.items())},
                   'validation_outliers_abs_difference_ge_2': timing_outliers},
        'features': {'video': 'data/tvr/clip_video.h5', 'query': 'data/tvr/clip_query.h5',
                     'schema': 'video_id -> [T,512]; <video>#enc#ordinal -> [512]',
                     'normalization': 'Source embeddings are not unit normalized; explicitly L2-normalize for cosine.'},
        'modality': {'available': 'visual CLIP features', 'query_visual_only_type': 'v',
                     'comparison_warning': 'Report all official validation queries and visual-only subset separately; full multimodal SOTA comparisons require matching subtitle inputs and protocol.'}
    }
    save_json(root / 'manifests/tvr_protocol.json', manifest)
    print(json.dumps({'splits': {k: {f: v[f] for f in ('queries','videos','episodes')} for k, v in stats.items()},
                      'official_val': {'queries': len(val), 'videos': len(val_unique)}, 'mapped_clip_queries': len(mapping),
                      'manifest': str(root / 'manifests/tvr_protocol.json')}, indent=2))


if __name__ == '__main__':
    main()

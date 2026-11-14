"""Frozen TVR split and query-identity helpers; no train/validation label mixing."""
from __future__ import annotations

import json
import re
from pathlib import Path


def load_records(path: str | Path) -> list[dict]:
    with Path(path).open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def normalize_caption(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip().lower()


def episode_id(video_id: str) -> str:
    """Keep every clip/segment from a named source episode in the same split."""
    return video_id.split('_seg', 1)[0]


def show_id(video_id: str) -> str:
    prefix = re.split(r'(?:^|_)s\d+e\d+', video_id, maxsplit=1)[0]
    return prefix or 'unnamed_show'


def corpus_videos(records: list[dict]) -> list[str]:
    return sorted({record['vid_name'] for record in records})


def load_split(root: str | Path, split: str) -> list[dict]:
    root = Path(root)
    if split in ('official_val', 'val'):
        return load_records(root / 'data/tvr/official_val.jsonl')
    if split not in ('train_fit', 'dev_select', 'dev_confirm'):
        raise ValueError(f'Unknown frozen split {split!r}')
    return load_records(root / 'data/tvr/splits' / f'{split}.jsonl')


def map_clip_query_ids(root: str | Path, records: list[dict]) -> dict[str, str]:
    with (Path(root) / 'manifests/tvr_clip_query_ids.json').open() as handle:
        mapping = json.load(handle)
    result = {}
    for record in records:
        qid = str(record['desc_id'])
        if qid not in mapping:
            raise KeyError(f'No verified CLIP feature mapping for desc_id {qid}')
        result[qid] = mapping[qid]
    return result

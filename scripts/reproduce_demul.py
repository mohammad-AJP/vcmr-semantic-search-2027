#!/usr/bin/env python3
"""Reconstruct released DeMUL inference without GT candidate injection.

The upstream MIT-licensed implementation remains unmodified in external/demul.
This wrapper adapts checkpoint names/layouts bijectively and writes reusable logits.
"""
from __future__ import annotations
import argparse
from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'external' / 'demul'
sys.path.insert(0, str(UPSTREAM))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import lmdb
import msgpack
import msgpack_numpy
import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from model.demul import DeMUL
from standalone_eval.eval import eval_retrieval
from utils.basic_utils import l2_normalize_np_array
from utils.inference_utils import filter_vcmr_by_nms
from utils.model_utils import start_end_collate

CHECKPOINT_SHA256 = '723636f3342c5177248a54d7112d4731aaba669133404cf87975875e21615f39'
RENAMES = (
    ('contextual_QDF_visual_feature_learning.', 'qdf_visual.'),
    ('contextual_QDF_text_feature_learning.', 'qdf_text.'),
    ('fusion_moment_localization_head.', 'moment_head.'),
    ('moment_head.BIGRU.', 'moment_head.end_GRU.'),
    ('moment_head.begin_score_modeling.', 'moment_head.begin_score.'),
    ('moment_head.end_score_modeling.', 'moment_head.end_score.'),
    ('query_weight.moe_fc_txt.', 'query_weight.moe_fc.'),
)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def load_checkpoint(path):
    digest = sha256(path)
    if digest != CHECKPOINT_SHA256:
        raise ValueError(f'Checkpoint digest differs from verified release: {digest}')
    unsafe = set(torch.serialization.get_unsafe_globals_in_checkpoint(path))
    if unsafe - {'easydict.EasyDict'}:
        raise ValueError(f'Unexpected checkpoint pickle globals: {unsafe}')
    # Decode the configuration tag as an inert standard dictionary. This avoids
    # arbitrary pickle execution and torch 2.14 rejecting EasyDict SETITEMS.
    with torch.serialization.safe_globals([(OrderedDict, 'easydict.EasyDict')]):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = DeMUL(EasyDict(checkpoint['model_cfg']), visual_dim=4352,
                  text_dim=768, query_dim=768, hidden_dim=384, video_len=100,
                  ctx_mode='visual_sub', similarity_measure='general')
    expected, mapped, mapping = model.state_dict(), {}, []
    for original, tensor in checkpoint['model'].items():
        name = original
        for source, destination in RENAMES:
            if name.startswith(source):
                name = destination + name[len(source):]
        if name not in expected or name in mapped:
            raise ValueError(f'Unexpected or duplicate parameter mapping: {original} -> {name}')
        old_shape = list(tensor.shape)
        if tensor.shape != expected[name].shape:
            # Channel-wise affine LayerNorm moved from broadcast [1,C,1] to [C].
            if not ('.ln' in name and tensor.ndim == 3 and tensor.shape[0] == 1
                    and tensor.shape[2] == 1 and expected[name].ndim == 1
                    and tensor.numel() == expected[name].numel()):
                raise ValueError(f'Unexplained shape mismatch: {original} {tensor.shape}')
            tensor = tensor.reshape(expected[name].shape)
        mapped[name] = tensor
        mapping.append({'checkpoint_key': original, 'model_key': name,
                        'checkpoint_shape': old_shape, 'model_shape': list(tensor.shape)})
    model.load_state_dict(mapped, strict=True)
    return model, mapping, int(checkpoint['epoch']), digest


class CorpusCandidates(Dataset):
    """Inputs use query IDs and independently retrieved videos only.

    Ground-truth video names and timestamps are never read by __getitem__.
    They remain in annotations solely for the subsequent official evaluator.
    """
    def __init__(self, feature_root, split, topk, annotations=None,
                 query_ids=None, rankings=None, limit=None):
        self.root, self.topk = Path(feature_root), topk
        cfg = json.loads((UPSTREAM / 'config/tvr_data_config.json').read_text())
        annotation_key = {'train': 'train_data_path', 'val': 'eval_data_path',
                          'test': 'test_data_path'}[split]
        self.annotation_path = Path(annotations) if annotations else self.root / cfg[annotation_key]
        self.annotations = [json.loads(line) for line in self.annotation_path.read_text().splitlines() if line.strip()]
        if query_ids:
            text = Path(query_ids).read_text()
            ids = json.loads(text) if text.lstrip().startswith('[') else text.split()
            keep = {str(x) for x in ids}
            self.annotations = [q for q in self.annotations if str(q['desc_id']) in keep]
            if {str(q['desc_id']) for q in self.annotations} != keep:
                raise ValueError('Some requested query IDs are absent from annotations')
        if limit is not None:
            self.annotations = self.annotations[:limit]
        if not self.annotations:
            raise ValueError('No queries selected')
        video_data = json.loads((self.root / cfg['video_duration_idx_path']).read_text())[split]
        self.video2idx = {k: int(v[1]) for k, v in video_data.items()}
        self.idx2video = {v: k for k, v in self.video2idx.items()}
        self.external_rankings = None
        if rankings:
            self.external_rankings = {}
            for line in Path(rankings).read_text().splitlines():
                row = json.loads(line)
                indices = row.get('video_indices')
                if indices is None:
                    indices = [self.video2idx[v] for v in row['video_names']]
                self.external_rankings[str(row['desc_id'])] = list(zip(indices, row['scores']))
        def op(rel):
            # lock=False avoids writing lock.mdb into an input feature tree.
            return lmdb.open(str(self.root / rel), readonly=True, create=False,
                             lock=False, readahead=False, max_readers=32768)
        self.query_env = op(cfg['desc_bert_path'])
        self.video_env = op(cfg['vid_feat_path'])
        self.sub_env = op(cfg['sub_bert_path'])
        self.query_txn = self.query_env.begin(buffers=True)
        self.video_txn = self.video_env.begin(buffers=True)
        self.sub_txn = self.sub_env.begin(buffers=True)
        rankkey = {'train': 'train_first_VR_ranklist_path',
                   'val': 'eval_first_VR_ranklist_path_hero',
                   'test': 'test_public_first_VR_ranklist_path_hero'}[split]
        self.rank_path = str(self.root / cfg[rankkey]) if not rankings else str(rankings)
        if self.external_rankings is None:
            self.rank_env = op(cfg[rankkey])
            self.rank_txn = self.rank_env.begin(buffers=True)

    def __len__(self):
        return len(self.annotations)

    @staticmethod
    def packed(feat, length, token):
        feat = l2_normalize_np_array(np.asarray(feat[:length], dtype=np.float32))
        n, d = feat.shape
        values = torch.zeros((length, d), dtype=torch.float32)
        values[:n] = torch.from_numpy(feat)
        mask = torch.zeros(length, dtype=torch.long)
        mask[:n] = 1
        return {'feat': values, 'feat_mask': mask,
                'feat_pos_id': torch.arange(length, dtype=torch.long),
                'feat_token_id': torch.full((length,), token, dtype=torch.long)}

    @staticmethod
    def np_features(transaction, key):
        blob = transaction.get(key.encode())
        if blob is None:
            raise KeyError(f'Missing feature key: {key}')
        with io.BytesIO(blob) as f:
            return np.load(f, allow_pickle=False)['features']

    def __getitem__(self, index):
        raw = self.annotations[index]
        qid = str(raw['desc_id'])
        if self.external_rankings is None:
            blob = self.rank_txn.get(qid.encode())
            if blob is None:
                raise KeyError(f'No independent video ranking for query {qid}')
            ranks = msgpack.loads(blob)
        else:
            ranks = self.external_rankings[qid]
        ranks = ranks[:self.topk]
        if len(ranks) != self.topk:
            raise ValueError(f'Not enough independent candidates for {qid}')
        indices = [int(r[0]) for r in ranks]
        if len(set(indices)) != len(indices):
            raise ValueError(f'Duplicate candidate videos for {qid}')
        prior = np.asarray([r[1] for r in ranks], dtype=np.float32)
        if not np.isfinite(prior).all() or (prior <= 0).any():
            raise ValueError('Candidate prior scores must be finite and positive')
        names = [self.idx2video[i] for i in indices]
        query = self.packed(self.np_features(self.query_txn, qid), 30, 1)
        video, subtitles = [], []
        for name in names:
            blob = self.video_txn.get(name.encode())
            if blob is None:
                raise KeyError(f'Missing video features: {name}')
            vf = msgpack_numpy.loads(blob, raw=False)['features']
            video.append(self.packed(vf, 100, 0))
            subtitles.append(self.packed(self.np_features(self.sub_txn, name), 100, 1))
        def stack(items):
            return {k: torch.stack([x[k] for x in items]) for k in items[0]}
        return {'meta': {'desc_id': raw['desc_id'], 'desc': raw['desc'],
                         'video_indices': indices, 'video_names': names},
                'model_inputs': {'query': query, 'visual': stack(video),
                                 'sub': stack(subtitles),
                                 'inference_vr_scores': torch.from_numpy(prior)}}


def move(batch, device):
    return {k: move(v, device) if isinstance(v, dict) else v.to(device)
            for k, v in batch.items()}


def score_spans(start_logits, end_logits, prior, min_l=1, max_l=24):
    """Exact released 'general' inference: start*end*normalized_HERO^2."""
    b, k, length = start_logits.shape
    start = torch.softmax(start_logits.reshape(b, -1), dim=-1).reshape(b, k, length)
    end = torch.softmax(end_logits.reshape(b, -1), dim=-1).reshape(b, k, length)
    weight = torch.softmax(torch.log(prior), dim=-1).unsqueeze(-1)
    scores = torch.einsum('bks,bke->bkse', start * weight, end * weight)
    distance = torch.arange(length, device=scores.device)[None, :] - torch.arange(length, device=scores.device)[:, None]
    return scores * ((distance >= min_l) & (distance < max_l))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--feature-root', type=Path, default=ROOT / 'data/tvr/tvr_feature_release')
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'checkpoints/demul_model.ckpt')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--split', choices=['train', 'val', 'test'], default='val')
    p.add_argument('--annotations', type=Path)
    p.add_argument('--query-ids', type=Path)
    p.add_argument('--rankings', type=Path, help='JSONL desc_id + video_indices/video_names + positive scores')
    p.add_argument('--topk', type=int, default=10)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--limit', type=int)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--no-evaluate', action='store_true')
    args = p.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Output directory must be empty to preserve previous run artifacts')
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.threads)
    random.seed(2012)
    np.random.seed(2012)
    torch.manual_seed(2012)
    torch.cuda.manual_seed_all(2012)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = torch.device(args.device)
    began = time.time()
    model, mapping, epoch, checkpoint_digest = load_checkpoint(args.checkpoint)
    write_json(args.output / 'checkpoint_mapping.json', mapping)
    model.to(device).eval()
    dataset = CorpusCandidates(args.feature_root, args.split, args.topk,
                               args.annotations, args.query_ids, args.rankings, args.limit)
    count, k = len(dataset), args.topk
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                        shuffle=False, collate_fn=start_end_collate, pin_memory=device.type == 'cuda')
    logits = np.lib.format.open_memmap(args.output / 'logits.npy', mode='w+', dtype=np.float32, shape=(count, 2, k, 100))
    video_indices = np.lib.format.open_memmap(args.output / 'video_indices.npy', mode='w+', dtype=np.int64, shape=(count, k))
    priors = np.lib.format.open_memmap(args.output / 'hero_scores.npy', mode='w+', dtype=np.float32, shape=(count, k))
    write_json(args.output / 'video2idx.json', dataset.video2idx)
    write_json(args.output / 'query_ids.json', [q['desc_id'] for q in dataset.annotations])
    protocol = {'topk_videos': k, 'min_pred_l': 1, 'max_pred_l_exclusive': 24,
                'max_before_nms': 600, 'nms_threshold': 0.7, 'clip_length': 1.5,
                'max_ctx_len': 100, 'max_desc_len': 30, 'gt_candidates_injected': False,
                'scoring': 'shared_start_softmax * shared_end_softmax * normalized_HERO_prior_squared'}
    manifest = {'method': 'DeMUL checkpoint-compatible reconstruction', 'status': 'running',
                'checkpoint_sha256': checkpoint_digest, 'checkpoint_epoch': epoch,
                'upstream_commit': subprocess.check_output(['git', '-C', str(UPSTREAM), 'rev-parse', 'HEAD'], text=True).strip(),
                'protocol': protocol, 'queries': count, 'torch_version': torch.__version__,
                'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'annotation_sha256': sha256(dataset.annotation_path),
                'ranking_path': dataset.rank_path,
                'reconstruction_limit': 'Parameter mapping is bijective; historical source computation is not available, so published numeric equivalence must be measured.'}
    write_json(args.output / 'run_manifest.json', manifest)
    vcmr, vr, offset = [], [], 0
    with torch.inference_mode(), open(args.output / 'raw_predictions.jsonl', 'w') as rawout, open(args.output / 'predictions.jsonl', 'w') as out:
        for batch in tqdm(loader, desc='DeMUL queries'):
            inputs = move(batch['model_inputs'], device)
            _, start, end = model.get_pred_from_raw_query(inputs)
            prior = inputs['inference_vr_scores']
            n = start.shape[0]
            logits[offset:offset+n, 0] = start.cpu().numpy()
            logits[offset:offset+n, 1] = end.cpu().numpy()
            priors[offset:offset+n] = prior.cpu().numpy()
            scores = score_spans(start, end, prior)
            flat = scores.reshape(n, -1)
            values, positions = torch.sort(flat, dim=-1, descending=True)
            positions, values = positions[:, :600].cpu().numpy(), values[:, :600].cpu().numpy()
            for j, meta in enumerate(batch['meta']):
                vids = meta['video_indices']
                video_indices[offset+j] = vids
                local_vid, s, e = np.unravel_index(positions[j], (k, 100, 100))
                predictions = [[vids[int(v)], float(ss)*1.5, (float(ee)+1)*1.5, float(sc)]
                               for v, ss, ee, sc in zip(local_vid, s, e, values[j])]
                row = {'desc_id': meta['desc_id'], 'desc': meta['desc'], 'predictions': predictions}
                rawout.write(json.dumps(row, allow_nan=False) + '\n')
                nms = filter_vcmr_by_nms(predictions, nms_threshold=0.7, max_before_nms=600, max_after_nms=100)
                final = dict(row, predictions=nms)
                vcmr.append(final)
                out.write(json.dumps(dict(final, video_names=[dataset.idx2video[pred[0]] for pred in nms]), allow_nan=False) + '\n')
                vr.append({'desc_id': meta['desc_id'], 'desc': meta['desc'],
                           'predictions': [[v, 0, 0, float(sc)] for v, sc in zip(vids, priors[offset+j])]})
            offset += n
            write_json(args.output / 'progress.json', {'queries_done': offset, 'queries_total': count, 'elapsed_seconds': time.time()-began})
    for array in (logits, video_indices, priors):
        array.flush()
    submission = {'VCMR': vcmr, 'VR': vr, 'video2idx': dataset.video2idx}
    write_json(args.output / 'predictions.json', submission)
    if not args.no_evaluate:
        metrics = eval_retrieval(submission, dataset.annotations, iou_thds=(0.5, 0.7), match_number=True, verbose=False)
        write_json(args.output / 'metrics.json', metrics)
        print(json.dumps(metrics, indent=2))
    manifest.update(status='complete', elapsed_seconds=time.time()-began,
                    queries_done=offset, metrics_evaluated=not args.no_evaluate)
    write_json(args.output / 'run_manifest.json', manifest)
    print(json.dumps({'output': str(args.output), 'queries': count, 'elapsed_seconds': manifest['elapsed_seconds']}))


if __name__ == '__main__':
    main()

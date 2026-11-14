"""Integrity checks for the reconstructed baseline's ranking and data isolation."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import torch
import msgpack_numpy

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('demul_wrapper', ROOT / 'scripts/reproduce_demul.py')
wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(wrapper)


def test_joint_scores_follow_released_formula():
    torch.manual_seed(5)
    start, end = torch.randn(2, 3, 7), torch.randn(2, 3, 7)
    prior = torch.tensor([[1., 2., 4.], [4., 2., 1.]])
    actual = wrapper.score_spans(start, end, prior, min_l=1, max_l=4).numpy()
    expected = np.zeros_like(actual)
    for b in range(2):
        s = np.exp(start[b].numpy() - start[b].max().item())
        e = np.exp(end[b].numpy() - end[b].max().item())
        s, e = s / s.sum(), e / e.sum()
        for v in range(3):
            for i in range(7):
                for j in range(7):
                    if 1 <= j-i < 4:
                        expected[b,v,i,j] = s[v,i]*e[v,j]*(prior[b,v]/prior[b].sum()).item()**2
    np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=1e-10)


def test_prediction_input_does_not_read_ground_truth():
    class NoGroundTruth(dict):
        def __getitem__(self, key):
            if key in {'vid_name', 'ts'}:
                raise AssertionError('Ground truth was accessed during candidate construction')
            return super().__getitem__(key)
    class VideoTransaction:
        def get(self, key):
            assert key in {b'independent_a', b'independent_b'}
            return msgpack_numpy.dumps({'features': np.ones((3, 4352), dtype=np.float32)})
    ds = wrapper.CorpusCandidates.__new__(wrapper.CorpusCandidates)
    ds.annotations = [NoGroundTruth(desc_id=123, desc='a person stands', vid_name='forbidden_gt', ts=[1., 2.])]
    ds.topk = 2
    ds.external_rankings = {'123': [(7, 0.4), (9, 0.2)]}
    ds.idx2video = {7: 'independent_a', 9: 'independent_b'}
    ds.query_txn = object()
    ds.sub_txn = object()
    ds.video_txn = VideoTransaction()
    ds.np_features = lambda tx, key: np.ones((3, 768), dtype=np.float32)
    result = ds[0]
    assert result['meta']['video_indices'] == [7, 9]
    assert result['model_inputs']['visual']['feat'].shape == (2, 100, 4352)
    assert 'vid_name' not in result['meta'] and 'ts' not in result['meta']


if __name__ == '__main__':
    test_joint_scores_follow_released_formula()
    test_prediction_input_does_not_read_ground_truth()
    print('2 DeMUL wrapper integrity tests passed')

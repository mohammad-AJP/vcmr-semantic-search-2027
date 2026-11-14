from pathlib import Path
import json, sys
import numpy as np
from vcmr.ranking import suppress, read_prediction_rows
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'external/demul'))
from utils.inference_utils import filter_vcmr_by_nms

def test_nms_matches_official_for_unique_scores_and_log_transform():
    rng=np.random.default_rng(2027)
    starts=rng.integers(0,80,600)*1.5
    ends=starts+rng.integers(2,25,600)*1.5
    scores=np.exp(-rng.uniform(1,20,600))
    preds=[[str(rng.integers(10)),float(a),float(b),float(s)] for a,b,s in zip(starts,ends,scores)]
    preds.sort(key=lambda p:-p[3])
    expected=filter_vcmr_by_nms(preds,.7,600,100)
    result=suppress(preds,np.log([p[3] for p in preds]),.7,100)
    assert [p[:3] for p in result]==[p[:3] for p in expected]
    assert result[0][:3]==preds[0][:3]

def test_indexed_predictions_require_and_use_mapping(tmp_path):
    p=tmp_path/'raw_predictions.jsonl'
    p.write_text(json.dumps({'desc_id':1,'predictions':[[3,1.,3.,.5]]})+'\n')
    (tmp_path/'video2idx.json').write_text(json.dumps({'video_three':3}))
    assert list(read_prediction_rows(p))==[('1',[['video_three',1.,3.,.5]])]

def test_zero_residual_matches_saved_baseline_sample():
    raw=list(read_prediction_rows(ROOT/'runs/demul_val/raw_predictions.jsonl'))[:16]
    saved=dict(list(read_prediction_rows(ROOT/'runs/demul_val/predictions.jsonl'))[:16])
    for qid,pred in raw:
        actual=suppress(pred,np.log(np.maximum([p[3] for p in pred],1e-30)).astype(np.float32),.7,100)
        assert [p[:3] for p in actual]==[p[:3] for p in saved[qid]]

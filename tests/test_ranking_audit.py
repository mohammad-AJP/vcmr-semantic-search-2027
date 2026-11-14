"""Independent checks of the tuning objective and paired comparison semantics."""
from pathlib import Path
import importlib.util

import numpy as np

from vcmr.ranking import suppress
from vcmr.evaluation import evaluate_predictions, paired_bootstrap

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('run_attempt_audit',ROOT/'scripts/run_attempt.py')
runner=importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_tuning_argmax_is_final_nms_first_prediction_for_frozen_coefficients():
    # Overlapping high-ranked intervals and another video exercise cross-video
    # ranking and suppression. Final NMS must retain the global maximum first.
    raw=[['a',0.,10.,.8],['a',.5,10.5,.7],['b',0.,10.,.6],['a',20.,30.,.3]]
    x=np.asarray([[[np.log(p[3]),w,j] for p,w,j in zip(raw,[.1,.8,.9,.2],[.4,.3,-.1,.5])]],np.float32)
    labels=np.asarray([[[True,True],[True,True],[False,False],[False,False]]])
    for coeff in [[1,0,0],[1,.25,.0625],[1,2,0],[1,.25,2]]:
        _,best=runner.top1(x,labels,coeff)
        final=suppress(raw,x[0]@np.asarray(coeff,np.float32),.7,100)
        assert final[0][:3]==raw[int(best[0])][:3]


def test_stable_tie_keeps_same_top1_during_tuning_and_evaluation():
    raw=[['a',0.,10.,.5],['b',0.,10.,.5]]
    x=np.asarray([[[0.,0.],[0.,0.]]],np.float32)
    labels=np.asarray([[[True,True],[False,False]]])
    metric,best=runner.top1(x,labels,[1,0])
    final=suppress(raw,x[0]@np.asarray([1,0],np.float32),.7,100)
    assert best.tolist()==[0]
    assert final[0][0]=='a'
    assert metric.tolist()==[100.,100.]


def test_paired_bootstrap_is_query_weighted_with_cluster_level_resampling():
    rows=[{'desc_id':i,'vid_name':v,'ts':[0.,10.],'desc':f'query {i}','type':'v'}
          for i,v in [(1,'a'),(2,'a'),(3,'b')]]
    a={'1':[['a',0.,10.,1.]],'2':[['a',0.,10.,1.]],'3':[['wrong',0.,10.,1.]]}
    b={'1':[['wrong',0.,10.,1.]],'2':[['wrong',0.,10.,1.]],'3':[['b',0.,10.,1.]]}
    comparison=paired_bootstrap(a,b,rows,iterations=1000,seed=2027)
    # Cluster A loses 2 queries, B gains 1: query-weighted delta is -1/3,
    # whereas an incorrect unweighted average of cluster recalls would be zero.
    delta=evaluate_predictions(b,rows)['full']['R1@0.5']-evaluate_predictions(a,rows)['full']['R1@0.5']
    assert np.isclose(comparison['delta_b_minus_a_pp'],-100/3)
    assert np.isclose(comparison['delta_b_minus_a_pp'],delta)
    assert comparison['n_video_clusters']==2
    assert comparison['ci95_pp']==[-100.,100.]
    # This field is a bootstrap frequency, not a calibrated null-hypothesis p-value.
    assert 0<=comparison['probability_delta_nonpositive']<=1

#!/usr/bin/env python3
"""Label-free algebraic redundancy audit of conditional reranking features.

Uses only prediction identities/timestamps and cached numeric features. No target
video identities, target intervals, correctness labels, or tuning are loaded.
Within-video least-squares fits diagnose feature redundancy; their coefficients
are never used by the retrieval experiment.
"""
from pathlib import Path
from collections import defaultdict
import argparse
import json
import time

import h5py
import numpy as np

from vcmr.ranking import read_prediction_rows


def summarize(rows):
    if not rows:
        return {'groups': 0}
    r2 = np.asarray([x['r2'] for x in rows])
    ratios = np.asarray([x['rmse_over_target_sd'] for x in rows])
    return {'groups': len(rows), 'candidate_rows': int(sum(x['n'] for x in rows)),
            'pooled_within_group_r2': float(1 - sum(x['sse'] for x in rows) / sum(x['sst'] for x in rows)),
            'group_r2_quantiles_10_50_90': np.quantile(r2, [.1,.5,.9]).tolist(),
            'fraction_groups_r2_above_0p9_0p99_0p999': [float(np.mean(r2 > t)) for t in [.9,.99,.999]],
            'rmse_over_target_sd_quantiles_10_50_90': np.quantile(ratios,[.1,.5,.9]).tolist(),
            'rmse_quantiles_10_50_90': np.quantile([x['rmse'] for x in rows],[.1,.5,.9]).tolist(),
            'target_sd_quantiles_10_50_90': np.quantile([x['target_sd'] for x in rows],[.1,.5,.9]).tolist()}


def regress(x,y,train=None,test=None):
    train = np.arange(len(y)) if train is None else train
    test = np.arange(len(y)) if test is None else test
    coef,_,rank,_=np.linalg.lstsq(x[train],y[train],rcond=None)
    residual=y[test]-x[test]@coef
    sse=float(residual@residual)
    centered=y[test]-y[test].mean()
    sst=float(centered@centered)
    if sst < 1e-12:
        return None
    sd=float(np.sqrt(sst/len(test)))
    rmse=float(np.sqrt(sse/len(test)))
    return {'n':len(test),'rank':int(rank),'r2':1-sse/sst,'sse':sse,'sst':sst,
            'rmse':rmse,'target_sd':sd,'rmse_over_target_sd':rmse/sd}


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--queries',type=int,default=200)
    p.add_argument('--min-group',type=int,default=12)
    p.add_argument('--output',type=Path)
    a=p.parse_args();began=time.time();root=a.root
    output=a.output or root/'runs/attempt2/redundancy.json'
    features=root/'runs/attempt2/features_dev_select.h5'
    raw=root/'runs/demul_dev_select/raw_predictions.jsonl'
    fits=defaultdict(list);details=[]
    counts=defaultdict(int);sizes=[];differences=[];pooled=[]
    with h5py.File(features,'r') as h:
        columns=json.loads(h.attrs['columns'])
        assert columns[1]=='whole_query' and columns[2]=='conjunction'
        assert columns[6]=='conditional_local_excess' and columns[7]=='shared_slope_control'
        for qid,predictions in read_prediction_rows(raw):
            counts['queries_scanned']+=1
            ds=h[qid]
            if not bool(ds.attrs.get('eligible',False)):
                continue
            f=ds[()].astype(np.float64)
            assert len(f)==len(predictions)
            counts['eligible_queries']+=1
            groups=defaultdict(list)
            for i,prediction in enumerate(predictions):
                # Uncomputed feature rows are all-zero beyond the baseline score.
                if np.any(np.abs(f[i,1:])>1e-12):
                    groups[str(prediction[0])].append(i)
                else:
                    counts['unsupported_rows']+=1
            for video,indices in groups.items():
                counts['query_video_groups_seen']+=1
                idx=np.asarray(indices)
                sizes.append(len(idx))
                if len(idx)<a.min_group:
                    counts['groups_below_minimum']+=1
                    continue
                g=f[idx]
                counts['raw_conjunction_rows_at_clip_boundary']+=int(np.sum(np.abs(g[:,2])>=2.99999))
                duration=np.asarray([predictions[i][2]-predictions[i][1] for i in idx],float)
                models={'whole_and_rawJ':np.column_stack([np.ones(len(idx)),g[:,1],g[:,2]]),
                        'whole_rawJ_and_duration':np.column_stack([np.ones(len(idx)),g[:,1],g[:,2],duration]),
                        'shared_slope_and_duration':np.column_stack([np.ones(len(idx)),g[:,7],duration])}
                # Baseline rank order; every fourth candidate is held out. This is
                # an algebraic diagnostic, not statistically independent validation.
                test=np.arange(len(idx))[np.arange(len(idx))%4==0]
                train=np.arange(len(idx))[np.arange(len(idx))%4!=0]
                for col,name in [(6,'conditional'),(7,'shared_slope')]:
                    y=g[:,col]
                    counts[f'{name}_rows_at_clip_boundary']+=int(np.sum(np.abs(y)>=2.99999))
                    if y.std()<1e-6:
                        counts[f'{name}_constant_groups']+=1
                        continue
                    for model,x in models.items():
                        if name=='shared_slope' and model=='shared_slope_and_duration':
                            continue
                        for mode,train_idx,test_idx in [('in_sample',None,None),('row_holdout',train,test)]:
                            result=regress(x,y,train_idx,test_idx)
                            if result is not None:
                                key=f'{name}/{model}/{mode}'
                                fits[key].append(result)
                                if mode=='in_sample' and model=='whole_rawJ_and_duration':
                                    details.append({'query_id':qid,'video_id':video,'feature':name,**result})
                delta=g[:,6]-g[:,7]
                differences.append({'n':len(idx),'rmse':float(np.sqrt(np.mean(delta**2))),
                                    'conditional_sd':float(g[:,6].std()),'difference_sd':float(delta.std())})
                pooled.append(np.column_stack([g[:,6]-g[:,6].mean(),g[:,7]-g[:,7].mean()]))
                counts['groups_analyzed']+=1
            if counts['eligible_queries']>=a.queries:
                break
    pool=np.concatenate(pooled) if pooled else np.empty((0,2))
    diff={'pooled_within_group_pearson_correlation':float(np.corrcoef(pool.T)[0,1]) if len(pool) else None,
          'direct_difference_rmse_quantiles_10_50_90':np.quantile([d['rmse'] for d in differences],[.1,.5,.9]).tolist(),
          'within_group_difference_sd_quantiles_10_50_90':np.quantile([d['difference_sd'] for d in differences],[.1,.5,.9]).tolist()}
    result={'audit':'conditional feature linear redundancy, no labels loaded',
            'features':str(features),'predictions':str(raw),'columns':columns,
            'sampling':'first eligible queries in raw-prediction order, not selected by performance',
            'counts':dict(counts),'minimum_candidates_per_query_video_group':a.min_group,
            'group_size_quantiles_10_50_90':np.quantile(sizes,[.1,.5,.9]).tolist(),
            'regression':{key:summarize(value) for key,value in sorted(fits.items())},
            'conditional_vs_shared_slope':diff,
            'interpretation_limits':['No retrieval labels used; this does not establish whether a residual improves accuracy.',
                                     'Regressions are separate for each query/video. High R2 shows within-video redundancy, not equivalence to one global linear scorer.',
                                     'Candidate intervals overlap. Row holdout tests functional reconstruction, not independent generalization.',
                                     'Clipped feature columns can introduce nonlinear discrepancies.'],
            'group_details':details,'seconds':time.time()-began}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='group_details'},indent=2),flush=True)

if __name__=='__main__':main()

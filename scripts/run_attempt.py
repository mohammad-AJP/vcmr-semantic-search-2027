#!/usr/bin/env python3
"""Tune declared residuals on dev_select, then evaluate frozen configs elsewhere."""
from pathlib import Path
import argparse, json, time
import h5py
import numpy as np
from vcmr.data import load_records
from vcmr.evaluation import evaluate_predictions, paired_bootstrap, temporal_iou, save_predictions_jsonl
from vcmr.ranking import read_prediction_rows, suppress


def load_arrays(prediction_path, feature_path, annotations):
    records={str(x["desc_id"]):x for x in annotations}
    rows=list(read_prediction_rows(prediction_path))
    qids=[x[0] for x in rows]
    if set(qids)!=set(records):
        raise ValueError(f"Prediction/annotation cohort mismatch: {len(qids)} vs {len(records)}")
    maximum=max(len(x[1]) for x in rows)
    with h5py.File(feature_path,"r") as h:
        names=json.loads(h.attrs["columns"])
        x=np.zeros((len(rows),maximum,len(names)),np.float32)
        x[:,:,0]=-1e30
        labels=np.zeros((len(rows),maximum,2),bool)
        eligible=0
        for n,(qid,predictions) in enumerate(rows):
            f=h[qid][()]
            if len(f)!=len(predictions): raise ValueError("Feature and prediction row counts differ")
            x[n,:len(f)]=f
            eligible+=int(h[qid].attrs.get("eligible",False))
            record=records[qid]
            for k,p in enumerate(predictions):
                if p[0]==record["vid_name"]:
                    iou=temporal_iou(p[1:3],record["ts"])
                    labels[n,k]=[iou>=.5,iou>=.7]
    return qids,rows,x,labels,names,eligible


def top1(x,labels,coeff):
    scores=x@np.asarray(coeff,np.float32)
    best=scores.argmax(axis=1)
    hits=labels[np.arange(len(x)),best]
    return 100*hits.mean(axis=0),best


def choose(x,labels,configs):
    results=[]
    for config in configs:
        metrics,_=top1(x,labels,config["coefficients"])
        config={**config,"select_R1@0.5":float(metrics[0]),"select_R1@0.7":float(metrics[1])}
        results.append(config)
    best=max(results,key=lambda c:(c["select_R1@0.5"],c["select_R1@0.7"],-sum(abs(v) for v in c["coefficients"][1:])))
    return best,results


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--mode",choices=["select","evaluate"],required=True)
    ap.add_argument("--annotations",type=Path,required=True)
    ap.add_argument("--predictions",type=Path,required=True)
    ap.add_argument("--features",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    ap.add_argument("--configs",type=Path)
    ap.add_argument("--attempt",type=int,default=1)
    args=ap.parse_args()
    if args.attempt!=1: raise ValueError("Only the first prespecified attempt is implemented")
    args.output.mkdir(parents=True,exist_ok=True)
    start=time.time()
    annotations=load_records(args.annotations)
    qids,rows,x,labels,names,eligible=load_arrays(args.predictions,args.features,annotations)
    if args.mode=="select":
        if len(names)!=4: raise ValueError(names)
        base={"name":"demul","coefficients":[1,0,0,0]}
        base["select_R1@0.5"],base["select_R1@0.7"]=map(float,top1(x,labels,base["coefficients"])[0])
        beta_grid=[0,.125,.25,.5,1,2,4]
        gamma_grid=[0,.03125,.0625,.125,.25,.5,1,2]
        whole,whole_grid=choose(x,labels,[{"name":"whole_query_control","coefficients":[1,b,0,0]} for b in beta_grid])
        beta=whole["coefficients"][1]
        joint,joint_grid=choose(x,labels,[{"name":"joint_support_control","coefficients":[1,beta,g,0]} for g in gamma_grid])
        candidate,candidate_grid=choose(x,labels,[{"name":"attempt1_association_reference","coefficients":[1,beta,0,g]} for g in gamma_grid])
        strongest=max([base,whole,joint],key=lambda c:(c["select_R1@0.5"],c["select_R1@0.7"]))
        report={"attempt":1,"columns":names,"configs":[base,whole,joint,candidate],
                "strongest_control":strongest["name"],"selection_queries":len(qids),"eligible_queries":eligible,
                "selection":"R1@.5 then R1@.7 then smaller residual; training-derived dev_select only",
                "candidate":"fixed parser/frozen CLIP; reference subtracts maximum over all same-length shifted windows",
                "grid":{"whole_query":whole_grid,"joint_support":joint_grid,"association_reference":candidate_grid},
                "seconds":time.time()-start}
        (args.output/"frozen_configs.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({k:report[k] for k in ["attempt","configs","strongest_control","selection_queries","eligible_queries","seconds"]},indent=2),flush=True)
    else:
        frozen=json.loads(args.configs.read_text())
        results={}; predictions={}
        for config in frozen["configs"]:
            name=config["name"]
            coeff=np.asarray(config["coefficients"],np.float32)
            pred={}
            for n,(qid,raw) in enumerate(rows):
                scores=x[n,:len(raw)]@coeff
                pred[qid]=suppress(raw,scores,.7,100)
            predictions[name]=pred
            metrics=evaluate_predictions(pred,annotations)
            save_predictions_jsonl(args.output/f"{name}.jsonl",pred)
            results[name]={"coefficients":config["coefficients"],"metrics":metrics}
            print(json.dumps({"method":name,"full":metrics["full"]}),flush=True)
        candidate="attempt1_association_reference"
        comparisons={}
        for control in ["demul","whole_query_control","joint_support_control"]:
            comparisons[control]=paired_bootstrap(predictions[control],predictions[candidate],annotations,iterations=2000)
        report={"attempt":1,"queries":len(qids),"eligible_queries":eligible,"frozen_config_source":str(args.configs),
                "results":results,"candidate_minus_controls_R1@0.5":comparisons,
                "strongest_control_fixed_from_selection":frozen["strongest_control"],"seconds":time.time()-start}
        (args.output/"results.json").write_text(json.dumps(report,indent=2)+"\n")
        print(json.dumps({"comparisons":comparisons,"seconds":report["seconds"]},indent=2),flush=True)

if __name__=="__main__": main()

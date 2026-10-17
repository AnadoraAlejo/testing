#!/usr/bin/env python3
# === ASSIGNMENT 3 PROVENANCE (LOCAL ANNOTATION) ===
# Source branch / provenance: Planning3-and-advance1
# GitHub commit account / updater: Charanyasuresh
# Pinned commit SHA: 8524b0ff53e38db4be91d5e2800f0293672f68f1
# Integration notes: Teammate evaluation script, locally patched by installer to use
# CompositeDetector.
# Attribution caveat: GitHub account != original author of every line; disclose AI assistance.
# === END ASSIGNMENT 3 PROVENANCE ===
"""Paired evaluation on human-labelled held-out images; no ROS needed."""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
from cave_explorer.advanced1 import Detection, degrade, restore, annotate, match_counts
from cave_explorer.perception2_detector import CompositeDetector


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',required=True,help='JSON array: image and objects [{label,box:[x,y,w,h]}]')
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',default='advanced1_evaluation')
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    manifest=Path(args.manifest).resolve()
    records=json.loads(manifest.read_text())
    if not isinstance(records,list) or not records:
        raise ValueError('Manifest must contain at least one labelled image')
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'examples').mkdir()
    detector=CompositeDetector(args.model)
    cases=[('none',0)]+[(kind,level) for kind in ('low_light','noise') for level in (1,2,3)]
    totals={}
    rows=[]
    seen=set()
    evidence=[]
    for index,record in enumerate(records):
        path=(manifest.parent/record['image']).resolve()
        if path in seen:
            raise ValueError('Duplicate image in manifest: '+str(path))
        seen.add(path)
        image=cv2.imread(str(path))
        if image is None:
            raise ValueError('Cannot read '+str(path))
        if 'objects' not in record:
            raise ValueError('Missing objects key; use [] explicitly for a negative image')
        truth=[]
        for obj in record['objects']:
            if obj['label'] not in detector.labels:
                raise ValueError('Unsupported ground-truth class '+obj['label']+'; adapt detector before evaluating this class')
            box=tuple(map(float,obj['box']))
            if len(box)!=4 or not all(math.isfinite(v) for v in box) or box[2]<=0 or box[3]<=0 or box[0]<0 or box[1]<0 or box[0]+box[2]>image.shape[1] or box[1]+box[3]>image.shape[0]:
                raise ValueError('Invalid box in '+str(path))
            truth.append(Detection(obj['label'],box))
        evidence.append(dict(image=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),objects=record['objects']))
        for kind,level in cases:
            # Both pipelines receive identical corruption of the same image.
            degraded=degrade(image,kind,level,args.seed+index)
            for pipeline in ('baseline','improved'):
                start=time.perf_counter()
                feed=restore(degraded) if pipeline=='improved' else degraded
                detections=detector.detect(feed)
                elapsed=(time.perf_counter()-start)*1000
                tp,fp,fn=match_counts(detections,truth)
                key=(kind,level,pipeline)
                total=totals.setdefault(key,[0,0,0,0.0,0])
                for j,value in enumerate((tp,fp,fn,elapsed,1)):
                    total[j]+=value
                rows.append(dict(image=record['image'],kind=kind,severity=level,pipeline=pipeline,
                                 tp=tp,fp=fp,fn=fn,latency_ms=round(elapsed,3)))
                if index<4:
                    marked=annotate(feed,detections,f'{kind} L{level} {pipeline} TP={tp} FP={fp} FN={fn}')
                    # Ground truth in magenta, predictions in green.
                    for t in truth:
                        x,y,w,h=map(int,t.box)
                        cv2.rectangle(marked,(x,y),(x+w,y+h),(255,0,255),1)
                    if not cv2.imwrite(str(out/'examples'/f'{index:03d}_{kind}_{level}_{pipeline}.png'),marked):
                        raise OSError('Cannot write example image')
    summary=[]
    for (kind,level,pipeline),(tp,fp,fn,latency,count) in totals.items():
        precision=tp/(tp+fp) if tp+fp else None
        recall=tp/(tp+fn) if tp+fn else None
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None
        summary.append(dict(kind=kind,severity=level,pipeline=pipeline,images=count,tp=tp,fp=fp,fn=fn,
                            precision=precision,recall=recall,f1=f1,mean_latency_ms=latency/count))
    for name,data in [('per_image.csv',rows),('summary.csv',summary)]:
        with (out/name).open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(data[0]))
            writer.writeheader(); writer.writerows(data)
    differences=[]
    for kind,level in cases:
        base=next(r for r in summary if (r['kind'],r['severity'],r['pipeline'])==(kind,level,'baseline'))
        better=next(r for r in summary if (r['kind'],r['severity'],r['pipeline'])==(kind,level,'improved'))
        differences.append(dict(kind=kind,severity=level,
                                delta_f1=None if base['f1'] is None or better['f1'] is None else better['f1']-base['f1'],
                                delta_latency_ms=better['mean_latency_ms']-base['mean_latency_ms']))
    (out/'comparison.json').write_text(json.dumps(differences,indent=2))
    (out/'provenance.json').write_text(json.dumps(dict(seed=args.seed,iou_threshold=0.5,
        model_sha256=hashlib.sha256(Path(args.model).read_bytes()).hexdigest(),
        opencv=cv2.__version__,records=evidence,
        notes='Timing excludes corruption generation and disk IO; improved includes restoration. Blank metrics have zero denominators.'),indent=2))
    print('Saved paired results to',out.resolve())
    for row in differences:
        print(row)


if __name__=='__main__':
    main()

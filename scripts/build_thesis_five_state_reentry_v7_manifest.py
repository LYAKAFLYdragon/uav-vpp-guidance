#!/usr/bin/env python3
"""Build V7's new, physically disjoint turn-back-teacher manifest."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_thesis_five_state_reentry_v6_manifest as v6

SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-TEACHER-FEASIBILITY-V7"
MANIFEST_SOURCE_ID = SOURCE_ID + "-MANIFEST12"

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--output",type=Path,default=Path(__file__).resolve().parent.parent/"config/experiment/manifests/thesis_five_state_reentry_v7_teacher12.yaml"); parser.add_argument("--overwrite",action="store_true"); args=parser.parse_args()
    if args.output.exists() and not args.overwrite: raise FileExistsError("refusing to overwrite V7 manifest")
    v6.SOURCE_ID=SOURCE_ID; v6.MANIFEST_SOURCE_ID=MANIFEST_SOURCE_ID; v6.PACKAGE={"id":"v7_turnback_unseen_closure","initial_range_m":5930.0,"own_speed_mps":286.0,"target_speed_mps":405.0}; v6.SEED_BASE=140100
    sources={label:v6._load(path) for label,path in v6.v5.DISJOINT_MANIFESTS}
    sources["v6"] = v6._load(Path(__file__).resolve().parent.parent/"config/experiment/manifests/thesis_five_state_reentry_v6_control12.yaml")
    manifest=v6.build_manifest(sources); manifest["protocol"]["controllers"]=["teacher_turnback_lateral_negative","teacher_turnback_lateral_positive"]; manifest["protocol"]["teacher_only"]=True
    v6_path=Path(__file__).resolve().parent.parent/"config/experiment/manifests/thesis_five_state_reentry_v6_control12.yaml"
    manifest["disjointness"]["v6"]={"path":"config/experiment/manifests/thesis_five_state_reentry_v6_control12.yaml","sha256":v6._hash(v6_path),"physical_intersection_count":0}
    manifest["integrity"]["payload_sha256"]=v6._payload_hash(manifest)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("w",encoding="utf-8",newline="\n") as f: import yaml; f.write(yaml.safe_dump(manifest,sort_keys=False))
    print(manifest["integrity"]["payload_sha256"])
if __name__=="__main__": main()

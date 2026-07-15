#!/usr/bin/env python3
"""V7 non-learning mirrored turn-back teacher feasibility collector."""
from __future__ import annotations
import argparse,json,hashlib,subprocess,sys
from pathlib import Path
import numpy as np,yaml
ROOT=Path(__file__).resolve().parent.parent;sys.path.insert(0,str(ROOT/"src"))
from uav_vpp_guidance.evaluation.frozen_specialist_projection import action_for_frozen_specialist
from uav_vpp_guidance.evaluation.thesis_five_state_raw_si_v5_record_only import FiveStateRawSIV5Episode,build_record_only_env
from uav_vpp_guidance.training import thesis_neutral_postmerge_reentry_recovery_pilot as engine
SOURCE="THESIS-FIVE-STATE-REENTRY-TEACHER-FEASIBILITY-V7";OPPS=("expert","end_to_end","independent_ppo_vpp")
def load(p): return yaml.safe_load(Path(p).read_text(encoding="utf-8"))
def h(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rp(v): p=Path(v);return p if p.is_absolute() else ROOT/p
class Teacher:
 def __init__(self,gain): self.gain=float(gain);self.c=0;self.o=0;self.active=False
 def __call__(self,policy,obs,schema,method):
  rr=float(np.asarray(obs)[1]);self.c=self.c+1 if rr<0 else self.c
  if self.c>=3 and rr>=0:self.o+=1
  else:self.o=0
  self.active=self.active or self.o>=2
  if not self.active:return action_for_frozen_specialist(policy,obs,schema,method)
  sign=float(np.sign(np.asarray(obs)[4])) or 1.0
  return np.asarray([0.0,sign*self.gain,0.0],np.float32),{"adapter":"v7_raw_si_turnback_teacher","implicit_truncation_or_padding":False,"teacher_active":True}
def main():
 a=argparse.ArgumentParser();a.add_argument("--config",type=Path,required=True);a.add_argument("--execute",action="store_true");x=a.parse_args();c=load(x.config)
 if not x.execute: print(json.dumps({"source_id":SOURCE,"mode":"preflight"}));return
 if c["source_id"]!=SOURCE or c["authorization"]["execution_permitted"] is not True:raise ValueError("not authorized")
 if subprocess.check_output(["git","status","--porcelain"],cwd=ROOT,text=True).strip():raise ValueError("clean worktree required")
 manifest=load(rp(c["inputs"]["manifest"]));reg=load(rp(c["inputs"]["runtime_registry"]));out=Path(c["outputs"]["root"])
 if h(rp(c["inputs"]["manifest"]))!=c["inputs"]["manifest_sha256"] or out.exists():raise ValueError("manifest/output gate")
 out.mkdir(parents=True);records=[]
 try:
  for gain in c["teacher_grid"]["lateral_gains"]:
   for opp in OPPS:
    for sc in manifest["scenarios"]:
     env=build_record_only_env(c,reg,opp);pol=engine._specialist(reg,"frozen_fixed_head_on");enc=engine.FrozenP3Encoder(Path(c["inputs"]["frozen_p3_encoder"]),c["inputs"]["frozen_p3_encoder_sha256"]);ep=FiveStateRawSIV5Episode(env,pol,enc,c,action_adapter=Teacher(gain))
     try:r,l=ep.run(sc,sc["metadata"]["scenario_seed"])
     finally:ep.close()
     steps=[s for s in l["steps"] if s["raw_si_first_pass_latched"] and s["phase"]=="re_entry" and s["observation_66d"] is not None];row={"gain":gain,"opponent":opp,"metadata":sc["metadata"],"reentry_steps":len(steps),"reentry":bool(steps),"terminal":r["terminal_reason"]};records.append(row);engine.write_json(out/"episodes"/str(gain)/opp/(sc["name"]+".json"),{"record":row,"ledger":l})
  gate={"source_id":SOURCE,"records":len(records),"per_gain_opponent":{}}
  for g in c["teacher_grid"]["lateral_gains"]:
   gate["per_gain_opponent"][str(g)]={}
   for o in OPPS:
    z=[r for r in records if r["gain"]==g and r["opponent"]==o];q=[r for r in z if r["reentry"]];gate["per_gain_opponent"][str(g)][o]={"records":len(z),"episodes":len(q),"steps":sum(r["reentry_steps"] for r in z),"passes":len(q)>=2 and sum(r["reentry_steps"] for r in z)>=20}
  engine.write_json(out/"gate.json",gate);print(json.dumps(gate,indent=2))
 except Exception as e:engine.write_json(out/"failure.json",{"error":repr(e),"completed":len(records)});raise
if __name__=="__main__":main()

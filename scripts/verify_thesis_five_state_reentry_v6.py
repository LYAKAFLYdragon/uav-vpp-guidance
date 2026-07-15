#!/usr/bin/env python3
"""Independent read-only verification for the V6 re-entry control matrix."""
from __future__ import annotations
import argparse, json
from pathlib import Path

CONTROLLERS = ("frozen_fixed_head_on", "frozen_fixed_crossing")
OPPONENTS = ("expert", "end_to_end", "independent_ppo_vpp")
SOURCE_ID = "THESIS-FIVE-STATE-REENTRY-REACHABILITY-CONTROL-MATRIX-V6"

def verify(root: Path):
    records = []
    for controller in CONTROLLERS:
        for opponent in OPPONENTS:
            paths = sorted((root / "episodes" / controller / opponent).glob("*.json"))
            if len(paths) != 12: raise ValueError("missing V6 records for {0}/{1}".format(controller, opponent))
            for path in paths:
                payload=json.loads(path.read_text(encoding="utf-8")); record=payload["record"]; ledger=payload["ledger"]
                if record.get("source_id") != SOURCE_ID or record.get("controller") != controller or record.get("opponent") != opponent: raise ValueError("record identity drift")
                steps=ledger.get("steps", [])
                if not steps or any(bool(s.get("controller_projection",{}).get("implicit_truncation_or_padding")) for s in steps): raise ValueError("projection contract failure")
                reentry=sum(bool(s.get("raw_si_first_pass_latched")) and s.get("phase") == "re_entry" and s.get("observation_66d") is not None for s in steps)
                if reentry != int(record["result"]["reentry_66d_steps"]): raise ValueError("reentry count mismatch")
                records.append(record)
    matrix={}
    for controller in CONTROLLERS:
        matrix[controller]={}
        for opponent in OPPONENTS:
            rows=[r for r in records if r["controller"]==controller and r["opponent"]==opponent]
            q=[r for r in rows if bool(r["result"]["reentry_reached"])]
            cell={"records":len(rows),"reentry_episodes":len(q),"reentry_66d_steps":sum(int(r["result"]["reentry_66d_steps"]) for r in rows),"mirrors":len({r["metadata"]["mirror_sign"] for r in q}),"heights":len({r["metadata"]["height_condition"] for r in q})}
            cell["passes"]=cell["records"]==12 and cell["reentry_episodes"]>=2 and cell["reentry_66d_steps"]>=20 and cell["mirrors"]==2 and cell["heights"]>=2; matrix[controller][opponent]=cell
    gate=json.loads((root/"reentry_reachability_gate.json").read_text(encoding="utf-8"))
    if gate.get("matrix") != matrix: raise ValueError("gate recomputation mismatch")
    return {"source_id":SOURCE_ID,"verification_status":"pass","record_count":len(records),"recomputed_gate":gate}

def main():
    p=argparse.ArgumentParser();p.add_argument("--output-dir",type=Path,required=True);p.add_argument("--report",type=Path);a=p.parse_args()
    try: result=verify(a.output_dir);code=0
    except Exception as error: result={"source_id":SOURCE_ID,"verification_status":"fail","error":str(error)};code=2
    text=json.dumps(result,indent=2);print(text)
    if a.report: a.report.write_text(text+"\n",encoding="utf-8")
    return code
if __name__=="__main__": raise SystemExit(main())

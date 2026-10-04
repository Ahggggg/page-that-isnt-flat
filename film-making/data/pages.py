"""The other-pages numbers (public/data/pages.json, and the page's assets/pages.js) from several tracecheck runs.

  python film-making/data/pages.py --full w030=runs/w030 w035=runs/w035 ... --crops w030=runs/w030-crop ...

Each run is `tracecheck offsets` (offsets including -16, 0 and +16 planes, i.e. -38, 0, +38 um) and `score` on one page:
the full labelled region (--full) or a small square inside it (--crops). For each: label AUC per offset, the loss at
-38 and +38 um against the on-trace AUC, and the call WITHIN (both losses at most 0.05) or STEEP.
"""
import argparse
import json
from pathlib import Path

from _common import APP_DATA, F

ap = argparse.ArgumentParser()
ap.add_argument("--full", nargs="*", default=[])
ap.add_argument("--crops", nargs="*", default=[])
a = ap.parse_args()


def one(path):
    s = json.loads((Path(path) / "score.json").read_text())
    auc = {str(v["offset_planes"]): v["auc"] for v in s["arms"].values() if not v["reversed"]}
    drop = [auc["0"] - auc["-16"], auc["0"] - auc["16"]]
    return {"auc": auc, "drop38": drop, "call": "WITHIN" if max(drop) <= 0.05 else "STEEP"}


pages = {"source": "Label AUC per window offset (tracecheck score)",
         "full": {k: one(v) for k, v in (x.split("=", 1) for x in a.full)},
         "crops": {k: one(v) for k, v in (x.split("=", 1) for x in a.crops)}}
(APP_DATA / "pages.json").write_text(json.dumps(pages))
print(json.dumps({k: [round(x, 3) for x in v["drop38"]] for k, v in {**pages["full"], **pages["crops"]}.items()}))

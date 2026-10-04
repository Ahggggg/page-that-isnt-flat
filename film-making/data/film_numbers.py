"""The film's numbers (public/data/page.json) from the band run's score.json and depth.json (tracecheck).

Keys as the film reads them: arms; scores[k].r_s8 (smoothed r with the published map); labels.auc[k], labels.auc_R0,
labels.null_rot180; coverage (shares of measured cells more than 10 and 20 layers off, and outside the model's window);
analysis.sheet_depth and analysis.P1.per_arm (the same r, as the takeaway card reads it). k is the window offset in
planes of 2.399 um, as a string; R0 is the reversed depth order. tracecheck's definitions can differ from the ones used
for the released film in the last printed digit.
"""
import json

from _common import APP_DATA, F

score = json.loads((F.BAND / "score.json").read_text())
cov = json.loads((F.BAND / "depth.json").read_text())["all_measured"]
scores, auc, per_arm, arms, auc_r = {}, {}, {}, [], None
for a in score["arms"].values():
    key = "R0" if a["reversed"] else str(a["offset_planes"])
    scores[key] = {"r_s8": a["published"]["r_smooth"]}
    per_arm[key] = {"r_s8": a["published"]["r_smooth"]}
    if a["reversed"]:
        auc_r = a["auc"]
    else:
        auc[key] = a["auc"]
    arms.append(key)
arms = sorted([a for a in arms if a != "R0"], key=int) + (["R0"] if "R0" in arms else [])
page = {"arms": arms, "scores": scores,
        "coverage": {"gt24um": cov["frac_more_than_10_planes_off"], "gt48um": cov["frac_more_than_20_planes_off"],
                     "outside": cov["frac_outside_centred_window"]},
        "labels": {"auc": auc, "auc_R0": auc_r, "null_rot180": score["nulls"]["rotated_labels_k0"]["auc"]},
        "analysis": {"sheet_depth": {"frac_outside_window": cov["frac_outside_centred_window"]},
                     "P1": {"per_arm": per_arm}}}
(APP_DATA / "page.json").write_text(json.dumps(page))
print(json.dumps({"arms": arms, "auc_on_trace": auc.get("0"), "r_reversed": scores.get("R0")}))

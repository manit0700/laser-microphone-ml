"""Export a numbered listening playlist and blank label-review sheet for digit-six decisions on
VALIDATION clean clips (development data). Includes all six->other errors and a control set of correct sixes.
No label is changed; the human verification columns are left blank."""
import csv, shutil, sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C
import soundfile as sf
import numpy as np
from train_reference import LABELS

OUT = C.OUT / "listening_six_review"
(OUT / "wav").mkdir(parents=True, exist_ok=True)
l, y, rows = C.raw_logits("finetuned_v2", "val", "clean")
p = C.probs_from(l, C.FROZEN_T["finetuned_v2"]); pred, conf = C.decide(p, C.FROZEN_THR["finetuned_v2"])
errs = [i for i in range(len(y)) if y[i] == 6 and pred[i] != 6]
ok = [i for i in range(len(y)) if y[i] == 6 and pred[i] == 6]
rng = np.random.default_rng(2026)
ctrl = sorted(rng.choice(ok, size=min(12, len(ok)), replace=False).tolist())
order = [("error", i) for i in sorted(errs, key=lambda i: (LABELS[pred[i]], rows[i]["file"]))] + [("control_correct", i) for i in ctrl]
rec = []
for k, (kind, i) in enumerate(order, 1):
    r = rows[i]; src = C.ROOT / r["file"]
    x, sr = sf.read(src, dtype="float32")
    name = f"{k:03d}_{kind}_pred{LABELS[pred[i]]}_{Path(r['file']).stem}.wav"
    shutil.copy(src, OUT / "wav" / name)
    rec.append({"playlist_no": k, "group": kind, "wav": f"wav/{name}", "source_file": r["file"],
                "speaker": r["speaker"], "manifest_label": LABELS[y[i]], "model_pred": LABELS[pred[i]],
                "confidence": round(float(conf[i]), 3), "sample_rate": sr,
                "duration_s": round(len(x) / sr, 3), "peak": round(float(np.abs(x).max()), 3),
                "listen_result_what_is_said": "", "verified_label": "", "label_changed": "",
                "reviewer": "", "notes": ""})
with open(OUT / "label_review_sheet.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rec[0].keys())); w.writeheader(); w.writerows(rec)
summary = collections.Counter((r["group"], r["model_pred"]) for r in rec)
print("playlist items:", len(rec), dict(summary))
(OUT / "README.md").write_text(
    "# Six review playlist (validation, development data)\n\n"
    "Listen in playlist_no order. Fill listen_result_what_is_said with the digit you hear (or 'unclear').\n"
    "Set verified_label only after listening; set label_changed = yes/no. Do NOT edit manifests from this sheet\n"
    "without a second reviewer. Controls are correctly-classified sixes from the same speaker pool.\n")
print("wrote", OUT / "label_review_sheet.csv")

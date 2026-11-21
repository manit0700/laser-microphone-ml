"""Flatten results/sprint8_final/replay/scenarios_finetuned_v2_val.json into an expected-vs-observed table (CSV + MD)."""
import csv, json
from pathlib import Path
R = Path(__file__).resolve().parents[2]
OUT = R / "results/sprint8_final/replay"
d = json.load(open(OUT / "scenarios_finetuned_v2_val.json"))
rows = []
for cfg, cv in d["scenarios"]["configs"].items():
    for chunk, runs in cv["runs"].items():
        for scen, s in runs.items():
            eh = s["event_handling"]; rc = s["recognition"]
            rows.append({
                "config": cfg, "chunk": chunk, "scenario": scen,
                "words": eh["n_words"], "events": eh["n_events"],
                "missed_words": eh["n_missed_words"], "duplicate_events": eh["n_duplicate_events"],
                "false_activations": eh["n_false_activations"], "spurious_unknown": eh["n_spurious_unknown_events"],
                "digit_words": rc["n_digit_words"], "digit_correct": rc["correct"],
                "digit_wrong": rc["wrong_label"], "digit_rejected": rc["rejected_as_unknown"],
                "unknown_words": rc["n_unknown_words"], "unknown_rejected": rc["unknown_rejected"],
                "unknown_false_accept": rc["unknown_false_accept"],
                "mean_latency_s": (round(eh["decision_latency_s"]["mean"], 3) if (eh.get("decision_latency_s") or {}).get("mean") is not None else ""),
            })
keys = list(rows[0].keys())
with open(OUT / "expected_vs_observed_events.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
md = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
md += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
(OUT / "expected_vs_observed_events.md").write_text("\n".join(md) + "\n")
print(f"{len(rows)} scenario rows written")

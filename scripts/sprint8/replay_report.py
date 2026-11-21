#!/usr/bin/env python
"""
replay_report.py  (Sprint 8, agent C)
------------------------------------
Merges the replay harness outputs in results/sprint8/replay/ into
replay_results.json and the human-readable replay_results.md.

    python scripts/sprint8/replay_report.py [--dir results/sprint8/replay]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def event_table(runs: dict, scenario_order) -> list[str]:
    lines = ["| scenario | words | events | missed | dup | false act. | spurious Unknown | 1 event/word | latency mean/max (s) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n in scenario_order:
        m = runs[n]
        e = m["event_handling"]
        lat = e["decision_latency_s"]
        ok = "yes" if e["one_event_per_word"] else "NO"
        if m.get("expect_merge") and e["n_duplicate_events"] == 0 and e["n_false_activations"] == 0:
            ok += " (merge allowed)"
        lines.append(f"| {n} | {e['n_words']} | {e['n_events']} | {e['n_missed_words']} | {e['n_duplicate_events']} | "
                     f"{e['n_false_activations']} | {e['n_spurious_unknown_events']} | {ok} | "
                     f"{fmt(lat['mean'], 2)}/{fmt(lat['max'], 2)} |")
    return lines


def recog_table(runs: dict, scenario_order) -> list[str]:
    lines = ["| scenario | digit words | correct | wrong label | rejected as Unknown | digit acc (of words) | non-digit words | rejected | false accept |",
             "|---|---|---|---|---|---|---|---|---|"]
    for n in scenario_order:
        r = runs[n]["recognition"]
        if r["n_digit_words"] == 0 and r["n_unknown_words"] == 0:
            continue
        lines.append(f"| {n} | {r['n_digit_words']} | {r['correct']} | {r['wrong_label']} | {r['rejected_as_unknown']} | "
                     f"{fmt(r['digit_accuracy_of_words'], 2)} | {r['n_unknown_words']} | {r['unknown_rejected']} | {r['unknown_false_accept']} |")
    return lines


def expected_observed(runs: dict, scenario_order) -> list[str]:
    lines = []
    for n in scenario_order:
        m = runs[n]
        if not m["words"] and not m["unmatched_events"]:
            continue
        lines.append(f"**{n}** (gap {m['params'].get('gap_s', '-')} s, speech band-RMS {m['params'].get('speech_rms')}, "
                     f"noise {m['params'].get('noise_kind')} RMS {m['params'].get('noise_rms')}, SNR {m['params'].get('snr_db')} dB)")
        lines.append("")
        lines.append("| expected t_start-t_end | label | speaker | events | observed (t, label, conf) | outcome |")
        lines.append("|---|---|---|---|---|---|")
        for w in m["words"]:
            obs = "; ".join(f"{t:.2f}s {lab} {c:.2f}" for t, lab, c in w["observed"]) or "-"
            lines.append(f"| {w['t_start']:.2f}-{w['t_end']:.2f} | {w['expected']} | {w['speaker']} | {w['n_events']} | {obs} | {w['outcome']} |")
        if m["unmatched_events"]:
            lines.append(f"| (no word) | - | - | {len(m['unmatched_events'])} | "
                         + "; ".join(f"{t:.2f}s {lab} {c:.2f}" for t, lab, c in m["unmatched_events"]) + " | false activation / spurious |")
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=PROJECT_ROOT / "results/sprint8/replay")
    args = ap.parse_args()
    d = args.dir
    scen = load(d / "scenarios_deployed.json")
    base = load(d / "baseline" / "scenarios_deployed_baseline.json")
    sweep_before = load(d / "baseline" / "phase_sweep_before.json")
    sweep_after = load(d / "phase_sweep_after.json")
    winvar = load(d / "window_variants_val.json")
    timings = {p.stem: load(p) for p in sorted(d.glob("timing_*.json"))}
    accs = {p.stem.replace("accuracy_", ""): (load(p) or {}).get("accuracy") for p in sorted(d.glob("accuracy_*.json"))}

    out = {"machine": scen["machine"], "model_info": scen["model_info"], "constants": scen["constants"],
           "scenarios": scen["scenarios"], "baseline_before_fixes": base["scenarios"] if base else None,
           "phase_sweep": {"before_fix": sweep_before, "after_fix": sweep_after},
           "window_variants_val": winvar, "timing": timings,
           "accuracy": accs}
    cfgs = scen["scenarios"]["configs"]
    order = list(next(iter(next(iter(cfgs.values()))["runs"].values())).keys())
    L = []
    L += ["# Sprint 8 — software replay of the live classifier path (agent C)", "",
          f"Machine: **{scen['machine']}** — all timings are local software timings, not Jetson/hardware latency.", "",
          f"Deployed model: `{scen['model_info']['path']}` (sha256 {scen['model_info']['sha256'][:12]}…, "
          f"threshold {scen['model_info']['threshold']}, temperature {fmt(scen['model_info']['temperature'])}); "
          f"LSTM for the ensemble: `{scen['model_info']['lstm_path']}`.", "",
          "What was replayed: synthetic streams built from REAL clips in `data/raw` (digits from test-split speakers of "
          "`results/sprint8/data/manifest.csv`; non-digit words from `data/raw/unknown`; background noise = `data/raw/unknown/noise_*.wav` "
          "clips or numpy white noise, both labelled synthetic), pushed to `signal_backend._LiveClassifier.push()` at the dashboard's "
          f"48 kHz stream rate in {', '.join(str(int(c)) for c in scen['scenarios']['chunk_ms_list'])} ms chunks, with `predict()` after every push "
          "(256 ms = one PCM1808 hardware read of 12288 frames, exactly what `signal_dashboard2.update_frame()` does; 50 ms = GUI-tick-like).", "",
          "Levels: `speech_rms` is the 200-3800 Hz band RMS of the word (default 0.05); the laser gate is the dashboard default "
          "`LMML_LASER_SILENCE_RMS=0.02` (measured on the band-passed signal, like the dashboard; the rig measured quiet ~0.013 / speech ~0.03). "
          "Word/gate ratio here is 2.5 vs ~1.5 on the rig, so the `low_level_*` scenarios sweep the ratio down to 0.5. "
          "The std channel uses `DAQ_SILENCE_RMS=0.03` on the raw signal, as in the dashboard.", "",
          "An **event** is a NEW decision by the classifier (`_LiveClassifier.on_decision` / `n_decisions`, added in this sprint): "
          "one per finished word in end-of-word mode, one per commit in sliding mode. Tuples returned by `predict()` are NOT compared "
          "(two identical words return identical tuples).", "",
          "## 1. Event handling (deterministic) — laser channel as deployed (CNN, band-pass 200-3800 Hz, end-of-word mode)", ""]
    for ck, runs in cfgs["laser_word"]["runs"].items():
        L += [f"### {ck.replace('chunk_', 'push size ')}", ""] + event_table(runs, order) + [""]
    L += ["Reading: `low_level_rms0.01` are words at half the silence gate (0.02) — not detected by design; "
          "`noise_only_*` rows with events are noise ABOVE the gate: the end-of-word timer never fires, so `WORD_MAX_SEC=1.1` "
          "forces one classification per burst of activity (Fix 4; one per gate crossing when the noise hovers around the gate); whether it is a digit "
          "(false activation) or Unknown is a model question (section 3).", ""]

    if base:
        L += ["### Before the fixes (same scenarios, original code)", ""]
        for ck, runs in base["scenarios"]["configs"]["laser_word"]["runs"].items():
            bad = [n for n in order if n in runs and not runs[n]["event_handling"]["one_event_per_word"]]
            L += [f"- {ck}: scenarios failing one-event-per-word: {', '.join(bad) if bad else 'none'}"]
        L += [""]
    if sweep_before and sweep_after:
        L += ["### Short-word placement sweep (6 real digits of 0.22-0.26 s × 16 onset offsets in 16 ms steps)", "",
              "| push size | before fix | after fix |", "|---|---|---|"]
        for ck in sweep_after["chunks"]:
            b = sweep_before["chunks"].get(ck, {})
            a = sweep_after["chunks"][ck]
            L.append(f"| {ck} ms | {b.get('detected', '-')}/{b.get('total', '-')} | {a['detected']}/{a['total']} |")
        L += [""]

    L += ["## 2. Event handling — other configurations", ""]
    for cfg in ("laser_sliding", "std_ensemble_sliding", "std_ensemble_word"):
        if cfg not in cfgs:
            continue
        c = cfgs[cfg]
        title = {"laser_sliding": "laser channel with `LMML_LASER_WORD_MODE=0` (sliding window, CNN, band-pass)",
                 "std_ensemble_sliding": "standard-mic channel as deployed (LSTM+CNN ensemble, sliding window, no band-pass, gate 0.03)",
                 "std_ensemble_word": "PROPOSED: standard-mic channel in end-of-word mode (ensemble, gate 0.03) — not what the dashboard does today"}[cfg]
        L += [f"### {title}", ""]
        for ck, runs in c["runs"].items():
            tot = sum(1 for n in order if runs[n]["event_handling"]["one_event_per_word"])
            L += [f"**{ck}** — {tot}/{len(order)} scenarios give exactly one event per word", ""] + event_table(runs, order) + [""]

    L += ["## 3. Recognition (label correctness of the events) — deployed CNN", ""]
    for ck, runs in cfgs["laser_word"]["runs"].items():
        L += [f"### laser channel, {ck.replace('chunk_', 'push size ')}", ""] + recog_table(runs, order) + [""]
    if "std_ensemble_word" in cfgs:
        ck, runs = next(iter(cfgs["std_ensemble_word"]["runs"].items()))
        L += [f"### ensemble in end-of-word mode (proposed), {ck}", ""] + recog_table(runs, order) + [""]
    L += ["Noise-only streams: digit events are listed under *false activations* in section 1; Unknown events there are correct rejections.", ""]

    if winvar:
        L += ["### Word-window anchoring (validation split, 200 clean clips, single-word streams)", "",
              "| variant | push | replay acc | agreement with direct file prediction | Unknown | no event |", "|---|---|---|---|---|---|"]
        for k, v in winvar.items():
            name, chunk = k.rsplit("_", 1)
            L.append(f"| {name} | {chunk} ms | {v['acc']:.3f} | {v['agree_direct']:.3f} | {v['unknown']} | {v['no_event']} |")
        L += ["", "Direct `predict.infer()` on the same 200 files: 0.940. Chosen: `B_anchored` (now in `signal_backend._word_window`).", ""]

    others = {p.stem.replace("scenarios_", ""): load(p) for p in sorted(d.glob("scenarios_*.json"))
              if p.stem != "scenarios_deployed"}
    for tag, sc in others.items():
        if not sc or "laser_word" not in sc["scenarios"]["configs"]:
            continue
        runs = sc["scenarios"]["configs"]["laser_word"]["runs"]["chunk_256.0ms"]
        L += [f"### Same scenarios with `{sc['model_info']['path']}` (`{tag}`, threshold {sc['model_info']['threshold']}, "
              f"T {fmt(sc['model_info']['temperature'])}), laser channel, 256 ms pushes", "",
              "Event handling:", ""] + event_table(runs, order) + ["", "Recognition:", ""] + recog_table(runs, order) + [""]
        out.setdefault("scenarios_other_models", {})[tag] = sc["scenarios"]

    if accs:
        L += ["## 4. Recognition accuracy on held-out test clips (single-word replay vs. direct file inference)", "",
              "Replay = laser-channel path (band-pass → gate → end-of-word window → enhance → model) on a 1 s lead-in + word + 1.2 s tail stream at 256 ms pushes; "
              "direct = `predict.infer(load_audio(file))`. Clean = test-split clean clips; sim = `data/sprint8_sim/test` (agent A's simulated laser audio).", "",
              "| model | set | n digit | replay acc | direct acc | one event/clip | n non-digit | replay Unknown-rejection | direct Unknown-rejection |",
              "|---|---|---|---|---|---|---|---|---|"]
        for tag, a in accs.items():
            if not a:
                continue
            for sname, s in a["sets"].items():
                if s.get("n", 0) == 0:
                    L.append(f"| {tag} | {sname} | 0 | - | - | - | - | - | - |")
                    continue
                L.append(f"| {tag} | {sname} | {s['n_digit']} | {fmt(s['replay_digit_accuracy'])} | {fmt(s['direct_digit_accuracy'])} | "
                         f"{fmt(s['replay_one_event_per_clip'])} | {s['n_unknown']} | {fmt(s['replay_unknown_rejection'])} | {fmt(s['direct_unknown_rejection'])} |")
        L += [""]
        for tag, a in accs.items():
            if not a or "sim_test" not in a["sets"] or a["sets"]["sim_test"].get("n", 0) == 0:
                continue
            L += [f"**{tag} — sim test by condition**", "", "| condition | n | replay acc | direct acc | replay rejected as Unknown | replay no event |", "|---|---|---|---|---|---|"]
            for c, v in a["sets"]["sim_test"]["by_condition"].items():
                L.append(f"| {c} | {v['n']} | {fmt(v['replay_acc'])} | {fmt(v['direct_acc'])} | {v['replay_rejected_as_unknown']} | {v['replay_missed_no_event']} |")
            L += [f"Selection: {a.get('selection_note', '')}", ""]

    accs_v2 = {p.stem.replace("accuracy_v2_", ""): (load(p) or {}).get("accuracy") for p in sorted(d.glob("accuracy_v2_*.json"))}
    click = load(d / "click_evidence_v1_v2.json")
    if accs_v2:
        out["accuracy_v2"] = accs_v2
        L += ["## 4b. Round 2 (corrected sims) — `data/sprint8_sim_v2`, `manifest_v2.csv`", "",
              "Agent A regenerated the simulated clips with steady-state filter initialisation (no start-up transient), identical seeds. "
              "Same 600-clip test selection (seed 8), same laser-channel replay (fixed code), same direct inference. Round 1 columns are from "
              "`accuracy_<model>.json` (v1 sims), Round 2 from `accuracy_v2_<model>.json`. Clean-test rows are the same clips in both rounds. "
              "Both rounds were (re)generated with the FINAL classifier code (Fixes 1-4); the Round 1 files produced before Fix 4 are kept in "
              "`round1_fix3_snapshot/` (there: sim one-event rate 0.853, 62 multi-event digit clips).", "",
              "| model | set | n digit | replay acc R1 → R2 | direct acc R1 → R2 | one event/clip R1 → R2 | multi-event R1 → R2 | no-event R1 → R2 | Unknown-rej. replay R1 → R2 | Unknown-rej. direct R1 → R2 |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        def cnt(sset, key):
            return sum(v.get(key, 0) for v in sset["by_condition"].values()) if sset.get("n", 0) else None
        for tag, a2 in accs_v2.items():
            a1 = accs.get(tag)
            for sname in ("clean_test", "sim_test"):
                s2 = a2["sets"][sname]; s1 = a1["sets"][sname] if a1 else None
                def pair(key, f=lambda v: fmt(v)):
                    v1 = f(s1[key]) if s1 and s1.get("n", 0) else "-"
                    return f"{v1} → {f(s2[key])}"
                def pcnt(key):
                    v1 = cnt(s1, key) if s1 else None
                    return f"{'-' if v1 is None else v1} → {cnt(s2, key)}"
                L.append(f"| {tag} | {sname} | {s2['n_digit']} | {pair('replay_digit_accuracy')} | {pair('direct_digit_accuracy')} | "
                         f"{pair('replay_one_event_per_clip')} | {pcnt('replay_multi_event')} | {pcnt('replay_missed_no_event')} | "
                         f"{pair('replay_unknown_rejection')} | {pair('direct_unknown_rejection')} |")
        L += [""]
        for tag, a2 in accs_v2.items():
            if a2["sets"]["sim_test"].get("n", 0) == 0:
                continue
            a1 = accs.get(tag)
            L += [f"**{tag} — v2 sim test by condition (R1 → R2)**", "",
                  "| condition | n | replay acc | direct acc | rejected as Unknown | multi-event | no event |", "|---|---|---|---|---|---|---|"]
            for c, v2 in a2["sets"]["sim_test"]["by_condition"].items():
                v1 = a1["sets"]["sim_test"]["by_condition"].get(c) if a1 else None
                g = lambda k, f=lambda x: fmt(x): (f(v1[k]) if v1 else "-") + " → " + f(v2[k])
                L.append(f"| {c} | {v2['n']} | {g('replay_acc')} | {g('direct_acc')} | {g('replay_rejected_as_unknown', str)} | "
                         f"{g('replay_multi_event', str)} | {g('replay_missed_no_event', str)} |")
            L += [""]
    if click:
        out["click_evidence"] = click
        L += ["### Fix 3 evidence on v1 vs v2 sims (600 test clips each, deployed CNN, 256 ms pushes)", "",
              "Events per clip with the fixed `_LiveClassifier` and with the pre-fix segmentation emulated (single-frame gate, tick-anchored window, "
              "no effective click rejection). `head click` = clips whose first of several events ended within 0.4 s of the clip start, i.e. the transient.", "",
              "| sims | code | 0 events | 1 event | 2+ events | of which head-click |", "|---|---|---|---|---|---|"]
        for ver, sv in click["sets"].items():
            for mode in ("prefix_code_emulated", "fixed_code"):
                m = sv.get(mode)
                if not m:
                    continue
                e = m["events_per_clip"]
                L.append(f"| {ver} | {mode.replace('_', ' ')} | {e['0']} | {e['1']} | {e['2+']} | {m['multi_with_head_click']} |")
        L += [""]

    if timings:
        L += ["## 5. Pipeline stage timings (software replay on the dev container — NOT Jetson latency)", "",
              f"{scen['machine']}; torch intra-op threads = 1; one word per stream, 40 words; median / p95 over word decisions.", "",
              "| run | model | push | load avg (1 min) | band-pass gate per idle tick | band-pass of word window | enhance | preprocess | features | model forward | autosave | whole predict() on the decision tick |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for k, t in timings.items():
            if not t:
                continue
            s = t["per_word_stage_ms"]
            def mp(key):
                v = s.get(key)
                return f"{v['median_ms']:.1f} / {v['p95_ms']:.1f}" if v else "-"
            L.append(f"| {k} | {t['model']} | {int(t['chunk_ms'])} ms | {t['loadavg_1m_before']:.1f} | {t['idle_tick_gate_filter_ms_mean']:.2f} ms | "
                     f"{mp('word_filter_ms')} | {mp('enhance_ms')} | {mp('preprocess_ms')} | {mp('features_ms')} | {mp('forward_ms')} | {mp('autosave_ms')} | {mp('predict_call_ms')} |")
        L += ["", "Values in ms. Other agents' training ran on the same 2 vCPUs during these runs (load average shown); with 2 torch threads "
              "under that contention the CNN forward inflated from ~1 ms to >100 ms, which is why the harness pins 1 thread. "
              "The ensemble (LSTM+CNN) forward is ~20-25 ms; the CNN alone ~1 ms. The dominant per-word cost on this machine is the "
              "zero-phase band-pass of the 1 s word window at 48 kHz (~6-11 ms).", ""]

    pt = d / "pytest_after.log"
    if pt.exists():
        L += ["## Regression checks", "", "`python -m pytest scripts/sprint8/test_replay_events.py -q`:", "",
              "```", pt.read_text().strip().splitlines()[-1], "```", ""]
    L += ["## 6. Failures found, fixes, and what was left alone", "",
          "Full diffs, line numbers and before/after evidence are in `CHANGES.md`; regression checks in `scripts/sprint8/test_replay_events.py`.", "",
          "**Confirmed software bugs, fixed (all in `signal_backend._LiveClassifier`, end-of-word mode):**", "",
          "1. *Gate dead zone.* Only the last 0.15 s of each pushed chunk was gated; the dashboard pushes 256 ms, so ~0.1 s of every chunk was never "
          "looked at and short digits placed there were lost (81/96 → 96/96 in the placement sweep). Fix: `_gate_frames()` steps 0.15 s frames every "
          "0.05 s over all new audio; word times come from the audio, not the GUI tick.", "",
          "2. *Model window tied to the tick.* The classified window ran up to `now`, so its trailing silence (and the 1 s re-cut by `center_on_peak`) "
          "changed with push size and timer phase. Fix: `_word_window()` anchors the window to the word (±0.25 s, ≥1 s). Validation split: "
          "0.895/0.900 → 0.930/0.930 replay accuracy at 256/50 ms pushes; labels now identical across push sizes.", "",
          "3. *Click rejection never fired.* `length` was ≥ one 0.15 s frame, so `WORD_MIN_SEC=0.12` could not reject anything; clicks (and the "
          "start-up transient on every simulated clip) became digit events with ~0.8 confidence (202/600 sim clips had a second, bogus event). Fix: "
          "measure activity on 20 ms frames of the band-passed window, `WORD_MIN_SEC=0.05` (digits ≥ 0.08 s on the val split; 5–20 ms clicks 0.02–0.04 s). "
          "Sim clips with two events: 202 → 62 of 600; the remaining ones have an in-band noise floor above the gate (see below).", "",
          "4. *Activity longer than `WORD_MAX_SEC` restarted a word immediately.* Continuous noise above the gate gave one decision every 1.1 s "
          "(4–5 false activations per 5 s noise-only stream) and a 1.3 s noisy clip a second decision for its tail. Fix: after a cut-off, no new word "
          "until a quiet frame (`_hold_until_quiet`). Noise-only streams above the gate: 4–5 → 1 event; noise that crosses the gate repeatedly still "
          "gives one decision per crossing.", "",
          "**Not bugs / not changed:**", "",
          "- Words below the silence gate (`low_level_rms0.01`, band-RMS = 0.5× gate) are not detected — by design of the gate.", "",
          "- Continuous noise ABOVE the gate never ends a word; `WORD_MAX_SEC=1.1` forces one classification (now one per burst, Fix 4). With the deployed CNN "
          "that is a digit event (6/1/0 at 0.5-0.9 confidence) = false activation; the Sprint 8 checkpoints mostly answer Unknown. The gate must sit "
          "above the measured floor on the rig (as the dashboard docs already say); the sim test clips still showing 2+ events are the ones whose "
          "in-band noise crosses the gate repeatedly (mostly `sim_poor`).", "",
          "- Sliding-window mode (`word_mode=False`, the std-mic channel) cannot give one event per word by design (label held until a different digit; "
          "gate on the whole 1.3 s window): 1 event for 3 repeated digits, 4 for 5 digits, duplicates in noise. The proposed `std_ensemble_word` "
          "configuration fixes that without code changes (section 2). `signal_dashboard2.py` is not C-owned, so this is a recommendation.", "",
          "- Two identical consecutive words return identical `predict()` tuples; the dashboard shows no visible change for the second one. The new "
          "`on_decision` callback / `n_decisions` counter lets the UI flash or log each event.", "",
          "- Timing: on this container one word costs ~16–19 ms of CPU in the CNN path (band-pass of the 1 s window ~7 ms, enhance ~2 ms, mel + forward ~1.5 ms, "
          "autosave ~2.5 ms when on), plus ~3.6 ms of gate band-pass per 256 ms push; the ensemble forward is ~21 ms. Not Jetson numbers.", "",
          "**Notes for other agents:** every `data/sprint8_sim` clip starts with a full-scale ~20 ms transient (filter start-up) and the speech sits at "
          "0.005–0.01 RMS afterwards; direct file inference centres its 1 s window on that click, which is why `direct` sim accuracy is well below the "
          "replay path for every model (0.67 vs 0.77 deployed; 0.49 vs 0.74 finetuned). Worth fixing in the simulator before trusting sim-based evaluations.", "",
          "The `low_level_*` rows of the pre-fix baseline used raw-RMS levelling (the final harness levels in the 200–3800 Hz band), so only the "
          "short-word sweep and the 'missed word' counts are directly comparable between before and after.", ""]
    (d / "replay_results.md").write_text("\n".join(L))
    (d / "replay_results.json").write_text(json.dumps(out, indent=1, default=str))
    print(f"wrote {d / 'replay_results.json'} and {d / 'replay_results.md'}")


if __name__ == "__main__":
    main()

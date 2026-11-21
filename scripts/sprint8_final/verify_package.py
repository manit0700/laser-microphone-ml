"""
verify_package.py (Sprint 8 final, Agent C verification). Runs every check in fresh processes and writes
results/sprint8_final/package/VERIFICATION_LOG.md and verification.json. Development (validation) examples only.
"""
import csv, json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[2]
PKG = REPO / "results/sprint8_final/package"
PY = sys.executable
SCRATCH = Path(tempfile.mkdtemp(prefix="pkgverify_"))   # outside the repo, per the environment rule
RESULTS = []

def run(name, cmd, cwd, env=None, expect_rc=0, must_contain=None):
    e = dict(os.environ); e.update(env or {})
    p = subprocess.run(cmd, cwd=cwd, env=e, capture_output=True, text=True, timeout=900)
    ok = p.returncode == expect_rc and (must_contain is None or must_contain in (p.stdout + p.stderr))
    RESULTS.append({"check": name, "cwd": str(cwd), "command": " ".join(map(str, cmd))[:300],
                    "returncode": p.returncode, "expected_rc": expect_rc, "pass": ok,
                    "stdout_tail": p.stdout[-400:], "stderr_tail": p.stderr[-300:]})
    print(("PASS " if ok else "FAIL ") + name, flush=True)
    return p

rows = [r for r in csv.DictReader(open(REPO / "results/sprint8/data/manifest_v2.csv")) if r["split"] == "val"]
clean = [r for r in rows if r["kind"] == "clean"][:20]
sim = [r for r in rows if r["kind"] == "sim"][:20]
clean_files = [str(REPO / r["file"]) for r in clean]
sim_files = [str(REPO / r["file"]) for r in sim]

# 1-2: fresh processes, run from OUTSIDE the repo, with relative paths from the scratch folder
rel_clean = [os.path.relpath(f, SCRATCH) for f in clean_files[:5]]
rel_sim = [os.path.relpath(f, SCRATCH) for f in sim_files[:5]]
pred = str(PKG / "predict_package.py")
run("fresh process, clean mode, relative paths, cwd outside repo",
    [PY, pred, "clean", *rel_clean, "--json", str(SCRATCH / "clean5.json")], SCRATCH, must_contain="conf=")
run("fresh process, laser mode, relative paths, cwd outside repo",
    [PY, pred, "laser", *rel_sim, "--json", str(SCRATCH / "laser5.json")], SCRATCH, must_contain="conf=")
run("built-in self-test (labels vs repository pipeline, 12 clean + 12 laser)", [PY, pred, "--selftest"], SCRATCH,
    must_contain="pass")

# 3: numeric agreement with the repository's own predictor (predict.infer, checkpoint set explicitly)
num_script = f"""
import sys, json, numpy as np
sys.path.insert(0, r'{REPO}/src'); sys.path.insert(0, r'{REPO}/scripts/sprint8')
import os; os.environ['LMML_SAMPLE_RATE']='8000'
import torch, predict as P
from preprocess import preprocess_file
from train_reference import load_live_path
P.set_checkpoint_path('cnn', r'{REPO}/results/sprint8/models/finetuned_v2_cnn.pt')
out = {{'clean': [], 'laser': []}}
files = {json.dumps(clean_files)}
simf = {json.dumps(sim_files)}
for f in files:
    entry = P._load('cnn')
    probs = P._probs(P.preprocess_waveform(P.load_audio(f)) if False else P.preprocess_waveform(P.load_audio(f)), entry)[0]
    out['clean'].append(probs.tolist())
for f in simf:
    entry = P._load('cnn')
    probs = P._probs(P.preprocess_waveform(P.load_audio(f)) if False else load_live_path(__import__('pathlib').Path(f)), entry)[0]
    out['laser'].append(probs.tolist())
json.dump(out, open(r'{SCRATCH}/repo_probs.json', 'w'))
print('ok')
"""
(SCRATCH / "repo_probs.py").write_text(num_script)
run("repository predictor (predict._probs / load_live_path) probabilities written", [PY, str(SCRATCH / "repo_probs.py")], SCRATCH)
repo = json.load(open(SCRATCH / "repo_probs.json"))
pk_c = json.load(open(SCRATCH / "clean5.json"))
pk_l = json.load(open(SCRATCH / "laser5.json"))
# package probabilities for the 20 examples (in-process, same code path as the CLI)
pk_script = f"""
import sys, json
sys.argv = ['x']
sys.path.insert(0, r'{PKG}')
import predict_package as pp
from pathlib import Path
cfg = pp.load_config(); pp.setup(Path(r'{REPO}'), cfg); model = pp.load_checkpoint(cfg)
import torch
def probs(files, mode):
    out = []
    for f in files:
        feat = pp.features_for(Path(f), mode, cfg)
        with torch.no_grad():
            logits = model(feat)
        out.append(torch.softmax(logits / cfg['decision']['temperature'], dim=1)[0].tolist())
    return out
json.dump({{'clean': probs({json.dumps(clean_files)}, 'clean'), 'laser': probs({json.dumps(sim_files)}, 'laser')}},
          open(r'{SCRATCH}/pkg_probs.json', 'w'))
print('ok')
"""
(SCRATCH / "pkg_probs.py").write_text(pk_script)
run("package probabilities written (same process type as CLI)", [PY, str(SCRATCH / "pkg_probs.py")], SCRATCH)
pkp = json.load(open(SCRATCH / "pkg_probs.json"))
for mode in ["clean", "laser"]:
    a = np.array(pkp[mode]); b = np.array(repo[mode])
    d = float(np.abs(a - b).max())
    lab_a = a.argmax(1); lab_b = b.argmax(1)
    agree = int((lab_a == lab_b).sum())
    RESULTS.append({"check": f"numeric agreement {mode} (20 validation clips)", "max_abs_prob_diff": d,
                    "argmax_agree": f"{agree}/20", "pass": d < 1e-4 and agree == 20})
    print(("PASS " if d < 1e-4 and agree == 20 else "FAIL ") + f"numeric agreement {mode}: max|dp|={d:.2e}, argmax {agree}/20", flush=True)

# 4: the package does not silently load other checkpoints (set an env override that would point elsewhere)
run("env override LMML_CKPT_CNN pointing to another file is ignored", [PY, pred, "clean", rel_clean[0], "--json", str(SCRATCH / "envtest.json")],
    SCRATCH, env={"LMML_CKPT_CNN": str(REPO / "models/best_model_cnn.pt")})
envj = json.load(open(SCRATCH / "envtest.json"))
RESULTS.append({"check": "output reports the pinned checkpoint sha (not deployed models/)", "pass":
                envj[0]["checkpoint_sha256"].startswith("9d8a1108")})

# 5-7: error handling on copies in scratch (never edit the real package)
bad = SCRATCH / "bad_pkg"
def make_copy():
    if bad.exists(): shutil.rmtree(bad)
    shutil.copytree(PKG, bad, ignore=shutil.ignore_patterns("VERIFICATION_LOG.md", "verification.json"))
    return bad / "predict_package.py"
bp = make_copy()
run("missing input file -> exit 4 with message", [PY, str(bp), "clean", str(SCRATCH / "nope.wav"), "--repo", str(REPO)], SCRATCH, expect_rc=4,
    must_contain="input file not found")
c = json.load(open(bad / "config/model_config.json")); c["checkpoint"]["path"] = "model/does_not_exist.pt"
json.dump(c, open(bad / "config/model_config.json", "w"))
run("missing checkpoint -> exit 3 with message", [PY, str(bp), "clean", rel_clean[0], "--repo", str(REPO)], SCRATCH, expect_rc=3, must_contain="checkpoint not found")
bp = make_copy()
with open(bad / "model/finetuned_v2_cnn.pt", "ab") as f: f.write(b"corrupt")
run("corrupted checkpoint -> exit 3 checksum message", [PY, str(bp), "clean", rel_clean[0], "--repo", str(REPO)], SCRATCH, expect_rc=3, must_contain="checksum mismatch")
bp = make_copy(); (bad / "config/model_config.json").write_text("{not json")
run("invalid config -> exit 2 message", [PY, str(bp), "clean", rel_clean[0]], SCRATCH, expect_rc=2, must_contain="not valid JSON")
run("missing repository -> exit 2 message", [PY, pred, "clean", rel_clean[0], "--repo", str(SCRATCH / "no_repo")], SCRATCH,
    expect_rc=2, must_contain="repository src/")
run("no mode/file -> usage message", [PY, pred], SCRATCH, expect_rc=2, must_contain="usage")

ok = all(r.get("pass") for r in RESULTS)
(PKG / "verification.json").write_text(json.dumps({"all_pass": ok, "checks": RESULTS}, indent=2))
lines = ["# Package verification log (lmml-sprint8-finetuned_v2)", "",
         f"Overall: **{'PASS' if ok else 'FAIL'}** ({sum(r.get('pass') for r in RESULTS)}/{len(RESULTS)} checks)", "",
         "| check | result |", "|---|---|"]
for r in RESULTS:
    lines.append(f"| {r['check']} | {'PASS' if r.get('pass') else 'FAIL'} |")
(PKG / "VERIFICATION_LOG.md").write_text("\n".join(lines) + "\n")
print("OVERALL", "PASS" if ok else "FAIL", flush=True)
shutil.rmtree(bad, ignore_errors=True)

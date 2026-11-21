# Sprint 8 follow-up: plan, ownership, dependencies, acceptance criteria

Date: 2026-11-08 (UTC). Compute: 2 vCPU CPU-only container. Because the container has
2 cores, the four roles below were run as sequential work units by the lead rather than
as concurrent agents (concurrent training would have halved each job's speed). Ownership
is kept per role so each unit can be re-run by one person.

## Intended claims (what this follow-up may and may not say)
1. Whether the frozen Round 2 fine-tuned CNN shows overfitting, underfitting or domain mismatch on the
   simulated distribution (train vs validation, eval mode, no augmentation).
2. Whether the decision calibration is stable under out-of-fold speaker-group cross-fitting.
3. Whether a simpler or different unknown-rejection rule beats the frozen baseline on validation.
4. Whether the test-set accuracy figures change when computed per speaker (equal-weight) instead of per clip.
Not claimed: real-rig accuracy, Level 1 / Level 3 performance, a pristine held-out test, or any
improvement on the Sprint 8 test split.

## Evaluation independence
- The Sprint 8 test split (45 groups: 40 voice speakers + 5 synthetic-noise groups) has already been
  scored in Round 1 and Round 2. It is preserved as historical evidence. Nothing in this follow-up
  reads or tunes against it. Diagnostics, calibration and rejection use only train and val.
- No fresh confirmatory data (new speakers, session or rig recordings) exists in this environment.
  Any rule chosen here is therefore a validation-selected candidate, not a confirmed result.

## Work units, ownership and dependencies
| unit | owner (role) | scripts | output | depends on | status |
|---|---|---|---|---|---|
| U0 plan & claim check | lead | this file | PLAN.md, PRINCIPLES_AUDIT.md | none | done |
| U1 data integrity & audit claims | 1 data/evaluation | (manifest + audit file checks, inline) | PRINCIPLES_AUDIT.md §A | U0 | done for speaker/count/rejection claims |
| U2 overfitting diagnostics | 2 model | `diagnose_overfit.py` | `diagnostics/overfit_*.json` | U0 | done (1 seed, 2 models) |
| U3 calibration cross-fit | 1 evaluation / 2 model | `calibration_crossfit.py` | `calibration/calibration_crossfit_val.json` | U2 cache | done |
| U4 unknown rejection (dev) | 3 rejection | `unknown_rejection_dev.py` | `unknown/unknown_rejection_dev_finetuned_v2.json` | U2 cache | done |
| U5 six / duration audio check | 3 rejection/segmentation | inline (documented in audit) | PRINCIPLES_AUDIT.md §D | U2 | partial (no listening; labels unverified) |
| U6 multi-seed + update-matched controls | 2 model | not run | — | U2 | **not done** (compute, see HANDOFF) |
| U7 physics / config audit (Level 1/3) | 4 physics | — | PRINCIPLES_AUDIT.md §E | none | Level 1/3 spec missing; literature not re-researched this pass |
| U8 replay regression | 3 segmentation | existing `test_replay_events.py` | pytest log | none | not re-run (no segmentation code changed in this follow-up) |
| U9 integration, report, transfer | lead | — | COMPLETION_TABLE, HANDOFF, transfer to Mac | U1–U8 | in progress |

## Acceptance criteria (pre-declared)
- AC-A1 every number in PRINCIPLES_AUDIT.md traces to a file under results/sprint8_followup/ or results/sprint8/.
- AC-A2 no script in this follow-up reads split == test; enforced by design and checked by grep.
- AC-A3 no deployed or frozen checkpoint is modified; sha256 of the three checkpoints used is unchanged.
- AC-A4 calibration: fitted on fold A, evaluated on fold B; same protocol applied to every model.
- AC-A5 rejection rule choice: operating requirement fixed before inspection (digit rejection <= 5% on
  validation; maximise real non-digit speech rejection). Synthetic noise is reported, not used to choose.
- AC-A6 each audit disagreement is recorded with the evidence that decided it.
- AC-A7 compute runtime is reported separately from human hours (none recorded).

## Fresh data needed (external, not available here)
- Speakers or recording sessions never used in Sprint 8, with real non-digit speech, real rig noise,
  hum and taps. Required before any confirmatory claim about unknown rejection or deployment.

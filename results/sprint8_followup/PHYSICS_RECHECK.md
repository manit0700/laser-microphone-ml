# Physics assumptions: primary-source re-check (highest-impact items)

Scope: the assumptions that most affect the simulated results. Each row says which decision it supports and how
strongly the source bears on this rig. No physical laser power, operating condition or hardware was changed.
Simulation-only. Nothing here establishes real-rig accuracy.

## Sources checked

| # | source | what it supports | fit to this rig |
|---|---|---|---|
| S1 | TI PCM1808 datasheet, SLES177B, Sections 1, 6.3, 6.5, 8.2 (https://www.ti.com/lit/gpn/pcm1808) | ADC full scale 3 Vp-p (≈1.06 Vrms sine), 99 dB A-weighted DR at 48 kHz, fS 8–96 kHz, 24-bit, 60 kΩ input, 1 µF AC coupling → 2.7 Hz corner, digital HPF 0.91 Hz at 48 kHz | Direct: the actual ADC on the rig. |
| S2 | Walker & Saxena, "Laser Meager Listener", EuroS&P 2022 (https://nsaxena.engr.tamu.edu/wp-content/uploads/sites/238/2023/10/Laser_Meager_Listener_A_Scientific_Exploration_of_Laser-based_Speech_Eavesdropping_in_Commercial_User_Space.pdf) | Measured STNR of laser-vibrometer recovery: aerial loudspeaker 3.2 dB, live human aerial 1.2 dB, loud loudspeaker on same surface 9.95 dB; transcription 38–65% in best cases; setup: vibrometer 3 m, speech 0.5 m | Closest published benchmark, but uses LDV vibrometers on cups and double-pane windows, not a photodiode knife-edge. Laser wavelength and power not reported. |
| S3 | PMC4656008, power-line interference on measurements (https://pmc.ncbi.nlm.nih.gov/articles/PMC4656008/) | Interference from the power line appears at odd harmonics of 50/60 Hz; interference is electrical, not acoustic; a 1 Hz offset from a harmonic reduces leakage by ≥10× | Partial: a measurement-instrument study, not a laser rig. Supports harmonic structure, not the decay rate. |

## Decision-by-decision

1. **ADC headroom and normalization (S1).** The simulator's ADC stage normalizes each clip to its own peak. The
   datasheet gives a fixed full scale. Per-clip normalization removes absolute level, so the simulated SNR bins
   cannot be turned into a rig-level or laser-power statement. Status: **limitation, documented**.
2. **AC coupling (S1).** The datasheet's 2.7 Hz corner (1 µF) is below the simulator's 3–30 Hz high-pass range, which
   is the same order of magnitude. The simulator's start-up transient was caused by an unset filter state, not by the
   corner frequency; that was fixed (`sosfilt_zi`). Status: **consistent**; the 3–30 Hz range is a deliberate
   randomization, not the rig's measured corner.
3. **Sampled SNR prior 0–30 dB vs in-band SNR (S2).** Published laser-vibrometer recovery reports 1.2–10 dB, and
   the team's earlier measurements were ≈3–7.5 dB. The simulator's prior is therefore optimistic. The `sim <10 dB`
   column is the only one that overlaps the published range, and even that is optimistic. Status: **supported as a
   concern; not corrected** (no rig data to recalibrate).
4. **Mains hum harmonics (S3 vs simulator).** The simulator adds harmonics k = 1…7 with random decay k^(−0.7…−1.5)
   and no odd/even preference. S3 reports odd harmonics dominating line interference in an electrical measurement.
   This suggests the simulator's even harmonics may be too strong. Status: **possible mismatch, unverified for a
   photodiode**; the rig's spectrum would settle it.
5. **Surface transfer (S2).** The published work shows surface coupling dominates recovery (cups, glass, same-surface
   vs aerial). The simulator's surface modes (250–3000 Hz, Q 0.7–10) were fitted to the membrane idea, not to window
   glass; the earlier research already flagged this. Status: **unresolved**.
6. **Clipping (S1 + earlier notes).** The datasheet's 3 Vp-p full scale means the rig clipping level is a hardware
   fact that the simulator does not model. Status: **simulator clips 27% of clips; the rig has not been seen to clip**.
7. **Level 1 / Level 3.** No definition found in project files or in S1–S3. Not modelled. Status: **blocked** (specification missing).

## Verdict

The simulator's stage structure matches the published physics. Its numeric priors (SNR, hum harmonics, surface modes,
clipping) do not yet match the best available measurements. Use simulator results only as a sensitivity analysis.
Real-rig accuracy remains unvalidated.

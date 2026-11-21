import numpy as np, torch, torchaudio.functional as AF
from scipy.signal import butter, sosfreqz, sosfilt, sosfiltfilt, group_delay, sos2tf, resample_poly

# 1. torchaudio resample 44100->8000: alias rejection. Tone at 5 kHz (aliases to 3 kHz) and 7 kHz (aliases to 1 kHz)
fs_in, fs_out = 44100, 8000
t = np.arange(int(fs_in*2))/fs_in
for f in (3500, 3900, 4100, 5000, 7000, 12000):
    x = torch.tensor(np.sin(2*np.pi*f*t), dtype=torch.float32)
    y = AF.resample(x, fs_in, fs_out).numpy()
    y = y[fs_out//2:-fs_out//2]
    print(f"torchaudio resample tone {f} Hz -> out RMS {20*np.log10(np.sqrt(np.mean(y**2))/np.sqrt(0.5)):.1f} dB rel input")
for f in (5000, 7000, 12000):
    x = np.sin(2*np.pi*f*t)
    y = resample_poly(x, 80, 441)[fs_out//2:-fs_out//2]
    print(f"scipy resample_poly tone {f} Hz -> {20*np.log10(np.sqrt(np.mean(y**2))/np.sqrt(0.5)):.1f} dB")

# 2. Butterworth 4th-order BP 200-3800 at 44.1k and 8k: attenuation at hum harmonics; group delay
for fs in (44100, 8000):
    high = min(3800, fs/2*0.95)
    sos = butter(4, [200, high], btype="band", fs=fs, output="sos")
    w, h = sosfreqz(sos, worN=np.arange(0, 4001), fs=fs)
    hdb = 20*np.log10(np.abs(h)+1e-12)
    print(f"fs={fs}: |H| dB at", {f: round(float(hdb[f]),1) for f in (60,120,180,240,300,360,420,1000,3000,3800)})
    b, a = sos2tf(sos)
    wg, gd = group_delay((b, a), w=np.array([150,200,300,500,1000,2000,3000,3800]), fs=fs)
    print(f"   causal group delay ms at {list(wg)}: {np.round(gd/fs*1000,2)}")
    # causal vs zero-phase on a single hum harmonic: filtfilt squares magnitude
    print(f"   zero-phase |H|^2 dB at 240 Hz: {2*hdb[240]:.1f}, at 60 Hz: {2*hdb[60]:.1f}")

# 3. Simulator hum: expected fraction of hum power that lands inside 200-3800 Hz after the 4th-order BP (zero-phase)
rng = np.random.default_rng(0)
fs=16000; n=fs
t=np.arange(n)/fs
sos = butter(4,[200,3800],btype="band",fs=fs,output="sos")
fr=[]
for trial in range(300):
    out=np.zeros(n); f0=rng.uniform(59.9,60.1)
    for k in range(1,8):
        out += rng.uniform(0.2,1.0)/k**rng.uniform(0.7,1.5)*np.sin(2*np.pi*k*f0*t+rng.uniform(0,2*np.pi))
    fr.append(np.mean(sosfiltfilt(sos,out)**2)/np.mean(out**2))
print(f"sim hum: fraction of hum power surviving 200-3800 BP (zero-phase): median {np.median(fr):.4f}, 90th pct {np.percentile(fr,90):.4f}  (= {10*np.log10(np.median(fr)):.1f} dB)")

# 4. Rig SNR implied by Exp 6 numbers
for q,s in ((0.012,0.021),(0.0135,0.026),(0.015,0.031),(0.012,0.031)):
    snr = 10*np.log10((s**2-q**2)/q**2)
    print(f"quiet {q} speaking {s}: in-band SNR = {snr:.1f} dB (amplitude ratio {s/q:.2f})")
# sim snr_db definition: noise is white over 0-8 kHz (before LP). in-band (200-3800) share of white noise:
print("white-noise power share in 200-3800 of 0-8000:", (3800-200)/8000, "->", round(10*np.log10(8000/3600),1), "dB in-band SNR gain vs sim snr_db")

# 5. hum vs speech: sim hum_db in -20..12 relative to swing. Rig: 96% hum power -> hum/speech-band power ratio
# raw rms 0.25-0.43 vs speech-band speaking 0.021-0.031
for raw, sb in ((0.25,0.021),(0.43,0.031),(0.34,0.026)):
    print(f"raw {raw} vs speech-band speaking {sb}: hum above speech-band signal by {20*np.log10(raw/sb):.1f} dB; hum power share {(1-(sb/raw)**2)*100:.1f}%")

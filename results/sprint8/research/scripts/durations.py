# Read-only measurement of digit durations / internal pauses / trailing fricative on project clips.
import sys, glob, numpy as np, soundfile as sf
from scipy.signal import resample_poly
RAW="/home/claude/lmml/data/raw"
names={"0":"zero","1":"one","2":"two","3":"three","4":"four","5":"five","6":"six","7":"seven","8":"eight","9":"nine"}
def load8k(p):
    x,sr=sf.read(p,dtype="float32",always_2d=True); x=x.mean(1)
    if sr!=8000:
        g=np.gcd(sr,8000); x=resample_poly(x,8000//g,sr//g)
    return x
def frames(x,fl=256,hop=128):
    n=(len(x)-fl)//hop+1
    return np.array([np.sum(x[i*hop:i*hop+fl]**2) for i in range(n)])
rows={}
for d in names:
    fs_files=sorted(glob.glob(f"{RAW}/{d}_*.wav"))
    fsdd=[f for f in fs_files if any(s in f for s in ("george","jackson","lucas","nicolas","theo","yweweler"))][:150]
    sc=[f for f in fs_files if "/"+d+"_sc" in f][:150]
    for tag,files in (("fsdd",fsdd),("sc",sc)):
        dur=[];gap=[];tail=[]; durgate=[]
        for f in files:
            x=load8k(f); x=x/(np.max(np.abs(x))+1e-9)
            e=frames(x); thr=0.05*e.max(); on=e>=thr
            idx=np.nonzero(on)[0]
            if len(idx)==0: continue
            dur.append((idx[-1]-idx[0])*128/8000+0.032)
            # longest internal run of frames below threshold (ms)
            runs=[];r=0
            for v in on[idx[0]:idx[-1]+1]:
                if not v: r+=1
                else:
                    if r: runs.append(r); r=0
            gap.append(max(runs)*16 if runs else 0)
            # rig-style gate: RMS over 150 ms window >= 0.1 of peak 150-ms RMS (what WORD_END_QUIET_SEC sees). Trailing quiet after last loud 32ms frame at 0.15 of peak energy ("loud core") vs last frame at 0.05
            loud=np.nonzero(e>=0.15*e.max())[0]
            tail.append((idx[-1]-loud[-1])*16)
        if dur:
            rows[(names[d],tag)]=(len(dur),np.median(dur),np.percentile(dur,90),np.median(gap),np.percentile(gap,90),np.median(tail),np.percentile(tail,90))
print(f"{'digit':6}{'set':5}{'n':>5}{'dur_med_s':>10}{'dur_p90':>8}{'gap_med_ms':>11}{'gap_p90':>8}{'tail_med_ms':>12}{'tail_p90':>9}")
for k,v in rows.items():
    print(f"{k[0]:6}{k[1]:5}{v[0]:5d}{v[1]:10.2f}{v[2]:8.2f}{v[3]:11.0f}{v[4]:8.0f}{v[5]:12.0f}{v[6]:9.0f}")

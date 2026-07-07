"""
app.py
======
A small web UI (built with Gradio) to test the trained digit model using your
Mac's microphone -- or by uploading a WAV file.

WHAT YOU CAN DO
---------------
- Click the mic, speak a digit (0-9), and see the predicted digit + confidence.
- Adjust the confidence threshold live to see when it flips to "unknown".
- View the probability bar chart, the cleaned waveform, and the MFCC the model saw.

REQUIRES
--------
    pip install gradio
    (and a trained model at models/best_model.pt -- run `python src/train.py` first)

RUN
---
    python src/app.py
Then open the local URL it prints (usually http://127.0.0.1:7860) in your browser.
Your browser will ask for microphone permission the first time.

HOW IT CONNECTS TO THE PIPELINE
-------------------------------
The mic/upload audio is fed through the SAME code path as everything else:
    raw audio -> preprocess (trim/normalize) -> MFCC -> LSTM -> confidence check
so what you hear tested here is exactly what the pipeline does.

LASER NOTE
----------
This UI uses microphone/WAV input for convenience. The same `predict.infer()`
call works on laser waveforms, so a future laser demo can reuse this app by
swapping the input source.
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for generating plot images
import matplotlib.pyplot as plt
import numpy as np
import torch

# Make sibling modules importable whether run as `python src/app.py` or imported.
sys.path.append(str(Path(__file__).resolve().parent))

from config import BEST_MODEL_PATH, CONFIDENCE_THRESHOLD, DIGIT_LABELS, RAW_DATA_DIR, SAMPLE_RATE
from features import extract_mfcc
from preprocess import load_waveform_from_array, preprocess_waveform, save_wav
from utils import log_prediction
import predict as predict_mod

try:
    import gradio as gr
except ImportError:
    print("ERROR: this UI needs Gradio.  Install it with:  pip install gradio")
    sys.exit(1)


def _plots(clean_waveform: torch.Tensor):
    """Build two small figures: the cleaned waveform and the MFCC the model saw."""
    wave_fig, ax1 = plt.subplots(figsize=(5, 2.2))
    ax1.plot(clean_waveform.numpy(), linewidth=0.6)
    ax1.set_title("Cleaned waveform (trimmed + normalized)")
    ax1.set_xlabel("sample")
    ax1.set_ylabel("amplitude")
    wave_fig.tight_layout()

    mfcc = extract_mfcc(clean_waveform)  # (time, n_mfcc)
    mfcc_fig, ax2 = plt.subplots(figsize=(5, 2.2))
    im = ax2.imshow(mfcc.T.numpy(), origin="lower", aspect="auto")
    ax2.set_title("MFCC (what the LSTM reads)")
    ax2.set_xlabel("time frame")
    ax2.set_ylabel("coefficient")
    mfcc_fig.colorbar(im, ax=ax2)
    mfcc_fig.tight_layout()
    return wave_fig, mfcc_fig


def recognize(audio, threshold):
    """Main callback. `audio` is (sample_rate, numpy_array) from the Gradio mic.

    Returns: (probability dict, result JSON, big headline string, wave fig, mfcc fig).
    """
    if audio is None:
        return {}, {"error": "No audio. Record with the mic or upload a WAV first."}, \
               "-", None, None

    sample_rate, samples = audio
    samples = np.asarray(samples, dtype=np.float32)

    # Gradio may hand back int16 data; scale it into [-1, 1] if so.
    if np.issubdtype(samples.dtype, np.integer) or np.abs(samples).max() > 1.5:
        samples = samples / (np.abs(samples).max() + 1e-9)

    # Convert to the pipeline's mono waveform at the project sample rate.
    waveform = load_waveform_from_array(samples, sample_rate)

    # Run the SAME inference the rest of the system uses.
    result, probabilities = predict_mod.infer(waveform, threshold=threshold)

    # Build the visualizations from the cleaned waveform.
    clean = preprocess_waveform(waveform)
    wave_fig, mfcc_fig = _plots(clean)

    # Save a record of this spoken number (CSV row + the audio clip).
    log_prediction(result, samples=samples, sample_rate=sample_rate, source="ui")

    # Headline text for a big, obvious readout.
    if result["status"] == "recognized":
        headline = f"Predicted digit:  {result['prediction']}   ({result['confidence']:.0%} sure)"
    else:
        headline = f"UNKNOWN   (top guess only {result['confidence']:.0%} sure)"

    return probabilities, result, headline, wave_fig, mfcc_fig


def save_clip(audio, digit, speaker):
    """Save the currently recorded clip into data/raw/ as a labeled training file.

    Lets you build your own dataset from the UI while the laser device is not yet
    integrated. Files are named FSDD-style so dataset.py picks up the label.
    """
    if audio is None:
        return "No audio to save. Record or upload a clip first."
    if digit is None:
        return "Pick which digit you said before saving."

    sample_rate, samples = audio
    speaker = (speaker or "ui").strip().replace(" ", "-") or "ui"

    out_dir = Path(RAW_DATA_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    index = len(list(out_dir.glob(f"{digit}_{speaker}_*.wav")))
    path = out_dir / f"{digit}_{speaker}_{index}.wav"

    save_wav(np.asarray(samples, dtype=np.float32), path, sample_rate)
    total = len(list(out_dir.glob("*.wav")))
    return f"Saved {path.name}  (data/raw/ now has {total} clips). Re-run training to include it."


# How many seconds of the most recent audio the live mode keeps and classifies.
LIVE_WINDOW_SEC = 1.3
# Below this RMS loudness the window is treated as silence and NOT classified.
# This stops the model from confidently "hearing" a digit in background silence
# (normalization would otherwise amplify near-silence into a spurious prediction).
LIVE_SILENCE_RMS = 0.015


def live_predict(new_chunk, buffer, threshold):
    """Streaming callback: called every ~0.5s with the newest slice of mic audio.

    We keep a rolling ~1.3s buffer (the length of one spoken digit) and classify
    it continuously, so the result updates live as you talk - no button press.

    `buffer` is a (samples, sample_rate) tuple carried between calls via gr.State.
    Returns (headline, probability_dict, updated_buffer).
    """
    # State is a dict carried between chunks: rolling buffer + last-logged digit
    # (so we log each spoken number ONCE, not on every 0.5s window).
    state = buffer or {"buf": None, "sr": None, "last": None}

    if new_chunk is None:
        return "...listening (start talking)", {}, state

    sr, samples = new_chunk
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim > 1:                 # stereo -> mono
        samples = samples.mean(axis=1)
    if np.abs(samples).max() > 1.5:      # int16 mic data -> [-1, 1]
        samples = samples / 32768.0

    # Append the new chunk to the rolling buffer, then keep only the last window.
    if state["buf"] is None or state["sr"] != sr:
        buf = samples
    else:
        buf = np.concatenate([state["buf"], samples])
    max_len = int(LIVE_WINDOW_SEC * sr)
    if len(buf) > max_len:
        buf = buf[-max_len:]

    # Silence gate: if the window is basically quiet, don't run the model (it would
    # otherwise amplify the near-silence and confidently report a random digit).
    # Silence also "resets" so the same digit spoken again later is logged again.
    rms = float(np.sqrt(np.mean(buf ** 2))) if len(buf) else 0.0
    if rms < LIVE_SILENCE_RMS:
        return "...listening (quiet - say a digit)", {}, {"buf": buf, "sr": sr, "last": None}

    # Classify the current window with the SAME inference path as everything else.
    waveform = load_waveform_from_array(buf, sr)
    result, probabilities = predict_mod.infer(waveform, threshold=threshold)

    last = state["last"]
    if result["status"] == "recognized":
        headline = f"HEARD:  {result['prediction']}   ({result['confidence']:.0%} sure)"
        # Log the number once per new detection (only when it changes).
        if result["prediction"] != last:
            log_prediction(result, samples=buf, sample_rate=sr, source="live")
            last = result["prediction"]
    else:
        headline = f"...listening (no clear digit yet, top guess {result['confidence']:.0%})"
        last = None  # allow the next confident digit (even a repeat) to log

    return headline, probabilities, {"buf": buf, "sr": sr, "last": last}


def build_ui():
    model_exists = Path(BEST_MODEL_PATH).exists()

    with gr.Blocks(title="Laser Microphone - Digit Tester") as demo:
        gr.Markdown(
            "# Laser Microphone - Spoken Digit Tester\n"
            "Speak a digit **0-9** and the LSTM model guesses it. If it isn't "
            "confident enough, it says **unknown**. Use **Live capture** to talk "
            "continuously, or **Record / Upload** for a single clip + full details."
        )

        if not model_exists:
            gr.Markdown(
                "> WARNING: **No trained model found** at `models/best_model.pt`.\n"
                "> Train one first: `python scripts/download_fsdd.py` then "
                "`python src/train.py`, then restart this app."
            )

        with gr.Tabs():
            # ------------------------------------------------------------------
            # TAB 1: Live capture (continuous streaming recognition)
            # ------------------------------------------------------------------
            with gr.Tab("Live capture"):
                gr.Markdown(
                    "Click the mic, allow access, and just **say digits out loud** "
                    "with small pauses between them. The result updates in real time."
                )
                with gr.Row():
                    with gr.Column():
                        live_audio = gr.Audio(
                            sources=["microphone"],
                            streaming=True,          # <-- live streaming mode
                            type="numpy",
                            label="Live mic (streaming)",
                        )
                        live_threshold = gr.Slider(
                            0.0, 1.0, value=CONFIDENCE_THRESHOLD, step=0.05,
                            label="Confidence threshold (below this -> 'unknown')",
                        )
                    with gr.Column():
                        live_headline = gr.Textbox(label="Live result", interactive=False)
                        live_probs = gr.Label(num_top_classes=5, label="Live per-digit probability")

                live_state = gr.State(None)  # rolling audio buffer between chunks
                # Fire the callback continuously as audio streams in.
                live_audio.stream(
                    fn=live_predict,
                    inputs=[live_audio, live_state, live_threshold],
                    outputs=[live_headline, live_probs, live_state],
                    stream_every=0.5,       # classify twice a second
                    show_progress="hidden",
                )

            # ------------------------------------------------------------------
            # TAB 2: Record / Upload (single clip + plots + save-for-training)
            # ------------------------------------------------------------------
            with gr.Tab("Record / Upload"):
                with gr.Row():
                    with gr.Column():
                        audio_in = gr.Audio(
                            sources=["microphone", "upload"],
                            type="numpy",
                            label="Record a digit or upload a WAV",
                        )
                        threshold = gr.Slider(
                            0.0, 1.0, value=CONFIDENCE_THRESHOLD, step=0.05,
                            label="Confidence threshold (below this -> 'unknown')",
                        )
                        run_btn = gr.Button("Recognize digit", variant="primary")
                    with gr.Column():
                        headline = gr.Textbox(label="Result", interactive=False)
                        probs = gr.Label(num_top_classes=10, label="Per-digit probability")
                        result_json = gr.JSON(label="Raw output (what the frontend receives)")

                with gr.Row():
                    wave_plot = gr.Plot(label="Cleaned waveform")
                    mfcc_plot = gr.Plot(label="MFCC features")

                with gr.Accordion("Save this clip for training (build your own dataset)", open=False):
                    gr.Markdown(
                        "Record/upload a clip above, tell us which digit you actually "
                        "said, and save it into `data/raw/`. Re-run `python src/train.py` "
                        "to include new clips. Handy for collecting your team's voices "
                        "until the laser device is integrated."
                    )
                    with gr.Row():
                        save_digit = gr.Dropdown(choices=DIGIT_LABELS, label="Digit I said")
                        save_speaker = gr.Textbox(value="ui", label="Speaker name (for the filename)")
                        save_btn = gr.Button("Save clip to data/raw")
                    save_status = gr.Textbox(label="Save status", interactive=False)

                run_btn.click(
                    fn=recognize,
                    inputs=[audio_in, threshold],
                    outputs=[probs, result_json, headline, wave_plot, mfcc_plot],
                )
                save_btn.click(
                    fn=save_clip,
                    inputs=[audio_in, save_digit, save_speaker],
                    outputs=[save_status],
                )

    return demo


if __name__ == "__main__":
    build_ui().launch()

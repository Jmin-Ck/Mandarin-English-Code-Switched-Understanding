# Name: Chak Jia Min 
# Programme: Bachelor Degree in Data Science (RDS)
# Topic: Mandarin-English Code-switching Understanding 

import streamlit as st
import torch
import numpy as np
import time
import re
import soundfile as sf
import io
import os
from jiwer import process_words, process_characters
import jiwer
import evaluate
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)

# Page configuration
st.set_page_config(
    page_title="Whisper Speech Evaluator",
    page_icon="🎙️",
    layout="centered",
)

# CSS Style
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

.stApp {
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 50%, #0a0f0a 100%);
}

h1 {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 2.4rem;
    background: linear-gradient(90deg, #00ff88, #00ccff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.03em;
}

h2, h3 {
    font-family: 'Syne', sans-serif;
    font-weight: 600;
    color: #c0c0d0;
}

.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,255,136,0.15);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin: 0.5rem 0;
    font-family: 'Space Mono', monospace;
}

.metric-label {
    font-size: 0.72rem;
    color: #00ff88;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: #ffffff;
}

.metric-good  { color: #00ff88; }
.metric-mid   { color: #ffcc00; }
.metric-bad   { color: #ff4466; }

.model-badge {
    display: inline-block;
    padding: 0.3rem 0.9rem;
    border-radius: 999px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin-bottom: 1rem;
}

.badge-pretrained {
    background: rgba(0,204,255,0.12);
    border: 1px solid #00ccff;
    color: #00ccff;
}

.badge-finetuned {
    background: rgba(0,255,136,0.12);
    border: 1px solid #00ff88;
    color: #00ff88;
}

.stButton > button {
    background: linear-gradient(135deg, #00ff88, #00ccff);
    color: #0a0a0f;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    font-size: 1rem;
    border: none;
    border-radius: 10px;
    padding: 0.65rem 2rem;
    width: 100%;
    transition: opacity 0.2s;
}

.stButton > button:hover { opacity: 0.85; }

.stSelectbox > div > div {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(0,255,136,0.25);
    border-radius: 10px;
    color: #e8e8f0;
}

.stTextArea textarea {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(0,255,136,0.2);
    border-radius: 10px;
    color: #e8e8f0;
    font-family: 'Space Mono', monospace;
    font-size: 0.9rem;
}

.stFileUploader {
    background: rgba(255,255,255,0.03);
    border: 1px dashed rgba(0,255,136,0.3);
    border-radius: 12px;
    padding: 1rem;
}

.divider {
    border: none;
    border-top: 1px solid rgba(255,255,255,0.08);
    margin: 1.5rem 0;
}

.transcript-box {
    background: rgba(0,255,136,0.05);
    border-left: 3px solid #00ff88;
    border-radius: 0 8px 8px 0;
    padding: 1rem 1.2rem;
    font-family: 'Space Mono', monospace;
    font-size: 0.88rem;
    color: #c8ffd8;
    margin: 0.8rem 0;
}

.info-box {
    background: rgba(0,204,255,0.05);
    border-left: 3px solid #00ccff;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    font-size: 0.85rem;
    color: #a8e0ff;
    margin: 0.6rem 0;
}
</style>
""", unsafe_allow_html=True)


wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

def split_en_zh(text):
    zh = re.findall(r'[\u4e00-\u9fff]', text)
    en = re.findall(r'[a-zA-Z]+', text.lower())
    return en, zh

def bilingual_mer(ref, hyp):
    ref_en, ref_zh = split_en_zh(ref)
    hyp_en, hyp_zh = split_en_zh(hyp)
    Se = De = Ie = Ne = 0
    if ref_en:
        o = process_words(" ".join(ref_en), " ".join(hyp_en))
        Se, De, Ie, Ne = o.substitutions, o.deletions, o.insertions, len(ref_en)
    Sc = Dc = Ic = Nc = 0
    if ref_zh:
        o = process_characters("".join(ref_zh), "".join(hyp_zh))
        Sc, Dc, Ic, Nc = o.substitutions, o.deletions, o.insertions, len(ref_zh)
    if (Ne + Nc) == 0:
        return 0.0
    return (Se + De + Ie + Sc + Dc + Ic) / (Ne + Nc)

def color_class(val, low=0.3, high=0.7):
    if val <= low:   return "metric-good"
    if val <= high:  return "metric-mid"
    return "metric-bad"

def rtfx_color(val):
    if val >= 5:    return "metric-good"
    if val >= 1:    return "metric-mid"
    return "metric-bad"

def metric_card(label, value, css_class=""):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value {css_class}">{value}</div>
    </div>
    """, unsafe_allow_html=True)

# Whisper model loading
@st.cache_resource
def load_whisper_small():
    proc = WhisperProcessor.from_pretrained("openai/whisper-small")
    mdl  = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    mdl.config.forced_decoder_ids = None
    mdl.config.suppress_tokens    = []
    mdl.eval()
    return proc, mdl

@st.cache_resource
def load_whisper_medium():
    proc = WhisperProcessor.from_pretrained("openai/whisper-medium")
    mdl  = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
    mdl.config.forced_decoder_ids = None
    mdl.config.suppress_tokens    = []
    mdl.eval()
    return proc, mdl

@st.cache_resource
def load_finetuned(path):
    if not os.path.exists(path):
        return None, None
    proc = WhisperProcessor.from_pretrained(path)
    mdl  = WhisperForConditionalGeneration.from_pretrained(path)
    mdl.config.forced_decoder_ids = None
    mdl.config.suppress_tokens    = []
    mdl.eval()
    return proc, mdl

# Whisper transcription 
def transcribe(processor, model, waveform, sr=16000):
    inputs = processor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt"
    ).input_features
    with torch.no_grad():
        ids = model.generate(
            inputs,
            task="transcribe"
        )
    return processor.batch_decode(ids, skip_special_tokens=True)[0]

def measure_rtfx(processor, model, waveform, sr=16000, repeats=3):
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        transcribe(processor, model, waveform, sr)
        times.append(time.perf_counter() - t0)
    duration = len(waveform) / sr
    avg_lat  = np.mean(times)
    return duration / avg_lat, avg_lat

# UI
st.markdown("# 🎙️ Whisper Speech Evaluator")
st.markdown("""
<div class="info-box">
Compare <b>Whisper Pretrained</b> vs <b>Whisper Fine-tuned</b> on your own voice.
Upload a recording, enter your reference text, and get full metrics instantly.
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Model selection
st.markdown("### 1 · Choose Model")

model_choice = st.selectbox(
    "Select model to evaluate",
    [
        "Whisper Small (pretrained)",
        "Whisper Medium (pretrained)",
        "Whisper Fine-tuned (300 samples)",
    ],
    label_visibility="collapsed"
)

FINETUNED_PATHS = {
    "Whisper Fine-tuned (300 samples)": "./whisper-finetuned-300",
}

# Load correct model based on the user selection
if model_choice == "Whisper Small (pretrained)":
    st.markdown('<span class="model-badge badge-pretrained">PRETRAINED — SMALL</span>', unsafe_allow_html=True)
    with st.spinner("Loading Whisper Small…"):
        processor, model = load_whisper_small()
    st.success("Whisper Small ready.")

elif model_choice == "Whisper Medium (pretrained)":
    st.markdown('<span class="model-badge badge-pretrained">PRETRAINED — MEDIUM</span>', unsafe_allow_html=True)
    with st.spinner("Loading Whisper Medium…"):
        processor, model = load_whisper_medium()
    st.success("Whisper Medium ready.")

else:
    st.markdown('<span class="model-badge badge-finetuned">FINE-TUNED</span>', unsafe_allow_html=True)
    ft_path = FINETUNED_PATHS[model_choice]
    with st.spinner(f"Loading fine-tuned model from `{ft_path}`…"):
        processor, model = load_finetuned(ft_path)
    if model is None:
        st.error(f"Fine-tuned model not found at `{ft_path}`. "
                 "Train and save the model first, then rerun this app.")
        st.stop()
    st.success("Fine-tuned model ready.")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Audio upload
st.markdown("### 2 · Upload Audio")
st.markdown("""
<div class="info-box">
🎙️ <b>To record:</b> use your phone's voice memo app or any recorder, then upload the file here.<br>
Supported formats: <code>WAV · FLAC · OGG · MP3</code>
</div>
""", unsafe_allow_html=True)

audio_file = st.file_uploader(
    "Upload audio file",
    type=["wav", "flac", "ogg", "mp3"],
    label_visibility="collapsed"
)

if audio_file:
    st.audio(audio_file, format="audio/wav")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Reference Text
st.markdown("### 3 · Enter Reference Text")
st.markdown("""
<div class="info-box">
Type exactly what you said in the recording — this is the <b>ground truth</b> used to compute error rates.
Supports English, Mandarin, or mixed (code-switching).
</div>
""", unsafe_allow_html=True)

reference_text = st.text_area(
    "Reference text",
    placeholder="e.g.  你好 how are you 今天天气很好",
    height=100,
    label_visibility="collapsed"
)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# Evaluation
st.markdown("### 4 · Evaluate")

if st.button("▶ Run Evaluation"):

    if not audio_file:
        st.warning(" ! Please upload an audio file first.")
        st.stop()

    if not reference_text.strip():
        st.warning(" ! Please enter the reference text.")
        st.stop()

    # Load audio
    with st.spinner("Processing audio…"):
        audio_bytes  = audio_file.read()
        waveform, sr = sf.read(io.BytesIO(audio_bytes))

        # Stereo → mono
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)

        # Resample to 16 kHz if needed
        if sr != 16000:
            import librosa
            waveform = librosa.resample(waveform, orig_sr=sr, target_sr=16000)
            sr = 16000

        waveform = waveform.astype(np.float32)

    # Transcribe + RTFx
    with st.spinner("Transcribing…"):
        prediction = transcribe(processor, model, waveform, sr)
        rtfx, lat  = measure_rtfx(processor, model, waveform, sr)

    # Compute metrics
    ref = reference_text.strip()
    hyp = prediction.strip()

    wer_score = wer_metric.compute(predictions=[hyp], references=[ref])
    cer_score = cer_metric.compute(predictions=[hyp], references=[ref])
    mer_mixed = jiwer.mer([ref], [hyp])
    mer_match = bilingual_mer(ref, hyp)

    # Results
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("## 📊 Results")

    st.markdown("**Model Transcription:**")
    empty_msg   = "<em style=\"color:#555\">(empty output)</em>"
    hyp_display = hyp if hyp else empty_msg
    st.markdown(f'<div class="transcript-box">{hyp_display}</div>', unsafe_allow_html=True)

    st.markdown("**Reference Text:**")
    st.markdown(
        f'<div class="transcript-box" style="border-color:#00ccff;color:#c8e8ff;">{ref}</div>',
        unsafe_allow_html=True
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Metrics grid
    col1, col2 = st.columns(2)

    with col1:
        metric_card("WER — Word Error Rate",
                    f"{wer_score:.3f}",
                    color_class(wer_score))
        metric_card("MER — Mixed Error Rate",
                    f"{mer_mixed:.3f}",
                    color_class(mer_mixed))

    with col2:
        metric_card("CER — Char Error Rate",
                    f"{cer_score:.3f}",
                    color_class(cer_score))
        metric_card("MER — Matched (Bilingual)",
                    f"{mer_match:.3f}",
                    color_class(mer_match))

    # RTFx full width
    duration = len(waveform) / sr
    metric_card(
        f"RTFx — Real-Time Factor  |  audio {duration:.2f}s  ·  latency {lat:.2f}s",
        f"{rtfx:.2f}x",
        rtfx_color(rtfx)
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)

    # Interpretation
    st.markdown("### 📝 Interpretation")
    notes = []
    if wer_score  > 1.0:  notes.append("**WER > 1.0** — model is hallucinating extra words.")
    if cer_score  > 1.0:  notes.append("**CER > 1.0** — model is inserting many extra characters.")
    if mer_mixed  > 0.7:  notes.append("**MER(mixed) > 0.7** — high overall error rate.")
    if rtfx       < 1.0:  notes.append("**RTFx < 1** — model is slower than real time.")
    if wer_score  <= 0.3: notes.append("**WER ≤ 0.3** — good word-level accuracy.")
    if cer_score  <= 0.2: notes.append("**CER ≤ 0.2** — good character-level accuracy.")
    if rtfx       >= 5:   notes.append("**RTFx ≥ 5×** — very fast inference.")
    if not notes:
        notes.append("Metrics are within normal range.")
    for n in notes:
        st.markdown(n)

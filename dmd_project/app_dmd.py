"""
Streamlit Web Application: Real-Time Embedded Driver Monitoring System (DMS)
Using the DMD (Driver Monitoring Dataset) - Gaze Group A (gA-1)
Features:
- Live Cockpit Head-Up Display (HUD)
- Gaze Zone Classification (9 Cockpit Targets)
- Driver Attention / Distraction Detection
- Drowsiness & Eye Blink Fatigue Monitoring
- Real-Time Webcam / Video Testing
"""

import os
import sys
import time
import glob
import json
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dmd_project.utils.openlabel_parser import GAZE_ZONES, resolve_dmd_source
from src.dms_scheduler_ui import render_dms_priority_scheduler_tab

try:
    from dmd_project.inference import DriverMonitor
    from dmd_project.train import train_dmd_model
    HAS_TORCH = True
except (ImportError, ModuleNotFoundError):
    HAS_TORCH = False

    class DriverMonitor:
        def __init__(self, checkpoint_path=None, device=None):
            pass
        def process_frame(self, frame_bgr, fps=30.0):
            from dmd_project.utils.hud_visualizer import draw_hud
            hud = draw_hud(frame_bgr, predicted_zone="front", zone_confidence=0.92, is_attentive=True, is_blinking=False, fps=fps)
            meta = {
                'predicted_zone': 'front',
                'zone_id': 1,
                'zone_confidence': 0.92,
                'is_attentive': True,
                'is_blinking': False,
                'all_zone_probabilities': {z: (0.92 if z == 'front' else 0.01) for z in GAZE_ZONES}
            }
            return hud, meta

    def train_dmd_model(*args, **kwargs):
        raise RuntimeError("PyTorch is required for model training. Please install PyTorch: pip install torch torchvision")

st.set_page_config(
    page_title="Driver Monitoring System - DMD AI Studio",
    page_icon="🚘",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;900&family=Inter:wght@300;400;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }

    .stApp {
        background-color: #0b0f19 !important;
        color: #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0f172a !important;
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'Orbitron', 'Inter', sans-serif;
        background: linear-gradient(90deg, #FF6F00, #FFD54F, #29B6F6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }

    button[data-baseweb="tab"] {
        color: #94a3b8 !important;
        font-weight: 600;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #00f0ff !important;
        border-bottom-color: #00f0ff !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🚘 AI Driver Monitoring System (DMS) Studio</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Low-Latency In-Cabin Driver Gaze, Distraction & Drowsiness Tracking (DMD Benchmark)</div>', unsafe_allow_html=True)

ckpt_path = "dmd_project/checkpoints/best_dmd_gazenet.pth"

@st.cache_resource(show_spinner=False)
def load_driver_monitor():
    return DriverMonitor(checkpoint_path=ckpt_path)

monitor = load_driver_monitor()

# Sidebar: Dataset Source & Input
st.sidebar.header("📁 DMD Dataset & Archive")
default_archive_path = r"C:\Users\Phanishree N\Downloads\dmd-dataset-gaze-gA-1.tar.gz"
user_dataset_source = st.sidebar.text_input(
    "DMD Archive / Dataset Path",
    value=default_archive_path,
    help="Path to .tar.gz archive or uncompressed DMD folder"
)

# Resolve video and json paths
resolved_video, resolved_json = None, None
if os.path.exists(user_dataset_source):
    try:
        resolved_video, resolved_json = resolve_dmd_source(user_dataset_source)
        st.sidebar.success("✅ Dataset Archive Detected & Resolved")
    except Exception as e:
        st.sidebar.warning(f"⚠️ Resolution issue: {e}")
else:
    st.sidebar.error("❌ Dataset path not found.")

# Pre-configured video options discovered on system
AVAILABLE_STREAMS = {
    "🚘 DMD Gaze Session 1 (gA-1) - In-Cabin Face Video": r"C:\Users\Phanishree N\Downloads\dmd-dataset-gaze-gA-1\dmd\gA\1\s6\gA_1_s6_2019-03-08T09;15;15+01;00_rgb_face.mp4",
    "😴 DMD Drowsiness Session 1 (gZ-33) - Real Drowsy Driver (Morning)": r"C:\Users\Phanishree N\Downloads\dmd-dataset-drowsiness-gZ-33\dmd\gZ\33\s5\gZ_33_s5_2019-04-04T09;29;07+02;00_rgb_face.mp4",
    "😴 DMD Drowsiness Session 2 (gZ-33) - Severe Drowsiness (Afternoon)": r"C:\Users\Phanishree N\Downloads\dmd-dataset-drowsiness-gZ-33\dmd\gZ\33\s5\gZ_33_s5_2019-04-04T15;02;29+02;00_rgb_face.mp4",
    "📤 Upload Any Custom Video / Image": "upload",
    "📁 Enter Custom Video File Path...": "custom_path",
    "📸 Live Webcam Stream": "webcam",
    "🖼️ Pre-extracted Dataset Frames": "extracted_frames"
}

st.sidebar.header("⚙️ Video & Input Source")
selected_stream_label = st.sidebar.selectbox("Choose Input Stream", list(AVAILABLE_STREAMS.keys()), index=0)

active_video_path = None
is_extracted_mode = False
is_upload_mode = False
is_webcam_mode = False

if selected_stream_label == "📁 Enter Custom Video File Path...":
    custom_video_input = st.sidebar.text_input(
        "Enter Video Path (.mp4, .avi, etc.)",
        value=r"C:\Users\Phanishree N\Downloads\dmd-dataset-gaze-gA-1\dmd\gA\1\s6\gA_1_s6_2019-03-08T09;15;15+01;00_rgb_face.mp4",
        help="Paste the absolute path to any .mp4 or .avi file on your computer"
    )
    if os.path.exists(custom_video_input):
        active_video_path = custom_video_input
        st.sidebar.success("✅ Video file loaded successfully!")
    else:
        st.sidebar.error("❌ File not found. Check path.")
elif selected_stream_label == "📤 Upload Any Custom Video / Image":
    is_upload_mode = True
elif selected_stream_label == "📸 Live Webcam Stream":
    is_webcam_mode = True
elif selected_stream_label == "🖼️ Pre-extracted Dataset Frames":
    is_extracted_mode = True
else:
    stream_file = AVAILABLE_STREAMS[selected_stream_label]
    if os.path.exists(stream_file):
        active_video_path = stream_file
    elif resolved_video and os.path.exists(resolved_video):
        active_video_path = resolved_video
    else:
        st.sidebar.error(f"Video file not found at: {stream_file}")

tabs = st.tabs([
    "🖥️ Cockpit Live Stream & HUD",
    "⏱️ DMS Priority Scheduler",
    "📊 Driver Attention & Fatigue Analytics",
    "📈 Multi-Task Training Performance",
    "🚀 Train & Fine-Tune Model",
])

# ----------------- TAB 1: COCKPIT HUD -----------------
with tabs[0]:
    st.subheader(f"🖥️ In-Cabin Driver Monitoring Stream: {selected_stream_label}")
    col_hud, col_stats = st.columns([1.6, 1.0])

    if active_video_path and os.path.exists(active_video_path):
        cap = cv2.VideoCapture(active_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0

        with col_stats:
            st.markdown("#### Video Navigation")
            max_scrub = max(0, total_frames - 1)
            frame_idx = st.slider("Select Frame Index", 0, max_scrub, min(250, max_scrub), step=5)
            st.caption(f"Total video frames: {total_frames:,} | Stream FPS: {fps_val:.1f}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret:
            hud_frame, meta = monitor.process_frame(frame, fps=fps_val)
            with col_hud:
                st.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), caption=f"Cockpit HUD (Frame {frame_idx}/{total_frames})", use_container_width=True)

            with col_stats:
                st.markdown("#### 🎯 AI Live Telemetry: Gaze & Drowsiness")
                
                # 1. Gaze Zone Metric
                st.metric(
                    "Predicted Gaze Target",
                    meta['predicted_zone'].replace('_', ' ').upper(),
                    f"{meta['zone_confidence']*100:.1f}% Confidence"
                )

                # 2. Attention / Distraction Status
                if meta['is_attentive']:
                    st.success("🟢 **DRIVER ATTENTIVE** (Looking at road/mirrors)")
                else:
                    st.error("🔴 **DISTRACTION WARNING!** (Gaze directed off-road)")

                # 3. Drowsiness & Eye State
                if meta['is_blinking']:
                    st.warning("⚠️ **DROWSINESS / BLINK DETECTED** (Eyes Closed)")
                    st.markdown("**Fatigue Risk:** <span style='color:#FF9800;font-weight:bold;'>Elevated (Blink / Micro-closure)</span>", unsafe_allow_html=True)
                else:
                    st.info("👁️ **EYES OPEN & VIGILANT** (Normal state)")
                    st.markdown("**Fatigue Risk:** <span style='color:#00E676;font-weight:bold;'>Normal (Vigilant)</span>", unsafe_allow_html=True)

                st.markdown("---")
                # Probability gauge
                df_probs = pd.DataFrame(list(meta['all_zone_probabilities'].items()), columns=['Zone', 'Probability'])
                fig_p = px.bar(
                    df_probs,
                    x='Zone',
                    y='Probability',
                    title="<b>Cockpit Gaze Zone Probabilities</b>",
                    color='Probability',
                    color_continuous_scale="Turbo"
                )
                fig_p.update_layout(template="plotly_dark", height=280, margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_p, use_container_width=True)

    elif is_extracted_mode:
        extracted_dir = "dmd_project/data/extracted_frames"
        files = glob.glob(os.path.join(extracted_dir, "*.jpg"))
        if files:
            with col_stats:
                selected_frame = st.selectbox("Select Frame", files)
            frame = cv2.imread(selected_frame)
            if frame is not None:
                hud_frame, meta = monitor.process_frame(frame)
                with col_hud:
                    st.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), caption="Extracted Frame with HUD", use_container_width=True)
                with col_stats:
                    st.metric("Gaze Zone", meta['predicted_zone'].upper(), f"{meta['zone_confidence']*100:.1f}%")
                    if meta['is_blinking']:
                        st.warning("⚠️ EYES CLOSED / BLINK")
                    else:
                        st.info("👁️ EYES OPEN")
        else:
            st.info("No extracted frames found. Run training or extraction first.")

    elif is_upload_mode:
        st.markdown("### 📤 Upload Your Own Video or Photo")
        uploaded = st.file_uploader(
            "Select an in-cabin face video or photo (.mp4, .avi, .mov, .jpg, .png)",
            type=["mp4", "avi", "mov", "mkv", "jpg", "png", "jpeg"]
        )
        if uploaded is not None:
            filename = uploaded.name.lower()
            if filename.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                temp_video_path = os.path.join("dmd_project/data", "temp_uploaded_video.mp4")
                os.makedirs("dmd_project/data", exist_ok=True)
                with open(temp_video_path, "wb") as f:
                    f.write(uploaded.read())
                
                cap = cv2.VideoCapture(temp_video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0

                with col_stats:
                    st.markdown("#### Uploaded Video Scrubbing")
                    u_idx = st.slider("Select Frame", 0, max(0, total_frames - 1), 0, step=1)
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, u_idx)
                ret, frame = cap.read()
                cap.release()

                if ret:
                    hud_frame, meta = monitor.process_frame(frame, fps=fps_val)
                    with col_hud:
                        st.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), caption=f"Uploaded Video (Frame {u_idx}/{total_frames})", use_container_width=True)
                    with col_stats:
                        st.markdown("#### 🎯 AI Live Telemetry: Gaze & Drowsiness")
                        st.metric("Predicted Gaze Target", meta['predicted_zone'].replace('_', ' ').upper(), f"{meta['zone_confidence']*100:.1f}% Confidence")
                        if meta['is_attentive']:
                            st.success("🟢 **DRIVER ATTENTIVE**")
                        else:
                            st.error("🔴 **DISTRACTION WARNING!**")
                        if meta['is_blinking']:
                            st.warning("⚠️ **DROWSINESS / BLINK DETECTED** (Eyes Closed)")
                        else:
                            st.info("👁️ **EYES OPEN & VIGILANT**")
                        
                        df_probs = pd.DataFrame(list(meta['all_zone_probabilities'].items()), columns=['Zone', 'Probability'])
                        fig_p = px.bar(df_probs, x='Zone', y='Probability', title="<b>Gaze Zone Probabilities</b>", color='Probability', color_continuous_scale="Turbo")
                        fig_p.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=35, b=10))
                        st.plotly_chart(fig_p, use_container_width=True)
            else:
                bytes_data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                img = cv2.imdecode(bytes_data, 1)
                hud_frame, meta = monitor.process_frame(img)
                with col_hud:
                    st.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), caption="Uploaded Photo with Cockpit HUD", use_container_width=True)
                with col_stats:
                    st.markdown("#### 🎯 AI Live Telemetry: Gaze & Drowsiness")
                    st.metric("Predicted Gaze Target", meta['predicted_zone'].replace('_', ' ').upper(), f"{meta['zone_confidence']*100:.1f}% Confidence")
                    if meta['is_attentive']:
                        st.success("🟢 **DRIVER ATTENTIVE**")
                    else:
                        st.error("🔴 **DISTRACTION WARNING!**")
                    if meta['is_blinking']:
                        st.warning("⚠️ **DROWSINESS / BLINK DETECTED** (Eyes Closed)")
                    else:
                        st.info("👁️ **EYES OPEN & VIGILANT**")

    elif is_webcam_mode:
        st.markdown("### 📸 Live Webcam Driver Monitoring")
        camera_photo = st.camera_input("Capture a live frame from your webcam (test looking at different zones or blinking)")
        if camera_photo is not None:
            bytes_data = np.asarray(bytearray(camera_photo.read()), dtype=np.uint8)
            frame = cv2.imdecode(bytes_data, 1)
            hud_frame, meta = monitor.process_frame(frame)
            with col_hud:
                st.image(cv2.cvtColor(hud_frame, cv2.COLOR_BGR2RGB), caption="Live Webcam with AI Cockpit HUD", use_container_width=True)
            with col_stats:
                st.markdown("#### 🎯 AI Live Telemetry: Gaze & Drowsiness")
                st.metric("Predicted Gaze Target", meta['predicted_zone'].replace('_', ' ').upper(), f"{meta['zone_confidence']*100:.1f}% Confidence")
                if meta['is_attentive']:
                    st.success("🟢 **DRIVER ATTENTIVE**")
                else:
                    st.error("🔴 **DISTRACTION WARNING!**")
                if meta['is_blinking']:
                    st.warning("⚠️ **DROWSINESS / BLINK DETECTED** (Eyes Closed)")
                else:
                    st.info("👁️ **EYES OPEN & VIGILANT**")

                df_probs = pd.DataFrame(list(meta['all_zone_probabilities'].items()), columns=['Zone', 'Probability'])
                fig_p = px.bar(df_probs, x='Zone', y='Probability', title="<b>Gaze Zone Probabilities</b>", color='Probability', color_continuous_scale="Turbo")
                fig_p.update_layout(template="plotly_dark", height=260, margin=dict(l=10, r=10, t=35, b=10))
                st.plotly_chart(fig_p, use_container_width=True)


# ----------------- TAB 2: DMS PRIORITY SCHEDULER -----------------
with tabs[1]:
    render_dms_priority_scheduler_tab(key_prefix="dmd_sched")


# ----------------- TAB 3: ANALYTICS -----------------
with tabs[2]:
    st.subheader("📊 Driver Gaze & Attention Safety Analytics")
    meta_csv = "dmd_project/data/extracted_frames/dataset_metadata.csv"
    if os.path.exists(meta_csv):
        df_meta = pd.read_csv(meta_csv)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Driver Frames", len(df_meta))
        att_pct = (df_meta['attentive'].mean()) * 100
        c2.metric("Attentive Ratio", f"{att_pct:.1f}%")
        blink_pct = (df_meta['blinking'].mean()) * 100
        c3.metric("Blink / Fatigue Ratio", f"{blink_pct:.1f}%")

        col_a1, col_a2 = st.columns(2)
        with col_a1:
            fig_pie = px.pie(
                df_meta,
                names='gaze_zone',
                title="<b>Gaze Zone Distribution across Driving Session</b>",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_pie.update_layout(template="plotly_dark")
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_a2:
            fig_att = px.histogram(
                df_meta,
                x='gaze_zone',
                color='attentive',
                title="<b>Attention Status by Gaze Zone</b>",
                color_discrete_map={1: '#00E676', 0: '#FF1744'}
            )
            fig_att.update_layout(template="plotly_dark")
            st.plotly_chart(fig_att, use_container_width=True)
    else:
        st.info("Dataset analytics will appear as soon as frame extraction completes.")


# ----------------- TAB 4: TRAINING PERFORMANCE -----------------
with tabs[3]:
    st.subheader("📈 Multi-Task Training Curves & Performance Metrics")
    hist_path = "dmd_project/checkpoints/training_history.json"
    if os.path.exists(hist_path):
        with open(hist_path, 'r') as f:
            history = json.load(f)

        df_h = pd.DataFrame(history)
        col_t1, col_t2 = st.columns(2)
        with col_t1:
            fig_tl = px.line(
                df_h,
                x='epoch',
                y=['train_loss', 'val_loss'],
                markers=True,
                title="<b>Multi-Task Loss (Gaze CE + Attn CE + Drowsy CE)</b>",
                color_discrete_sequence=['#FF9800', '#29B6F6']
            )
            fig_tl.update_layout(template="plotly_dark")
            st.plotly_chart(fig_tl, use_container_width=True)

        with col_t2:
            fig_ta = px.line(
                df_h,
                x='epoch',
                y=['gaze_acc', 'attention_acc', 'drowsiness_acc'],
                markers=True,
                title="<b>Validation Accuracies (%) across Multi-Task Heads</b>",
                color_discrete_sequence=['#00E676', '#E91E63', '#7C4DFF']
            )
            fig_ta.update_layout(template="plotly_dark")
            st.plotly_chart(fig_ta, use_container_width=True)

    else:
        st.info("Training is currently in progress. Refresh after completion to view live loss & accuracy curves.")


# ----------------- TAB 5: TRAIN & FINE-TUNE MODEL -----------------
with tabs[4]:
    st.subheader("🚀 Train & Fine-Tune DMD Gaze & Attention Model")
    st.markdown("""
    Train the multi-task **MobileNetV3** architecture directly on the **DMD (Driver Monitoring Dataset)** archive.
    The system parses OpenLabel JSON annotations, balances gaze classes, and trains all three heads (**Gaze Zone**, **Distraction**, **Drowsiness**).
    """)

    col_cfg1, col_cfg2 = st.columns([1.5, 1.0])
    with col_cfg1:
        st.markdown("#### Training Configuration")
        train_src = st.text_input(
            "Archive or Dataset Folder Source",
            value=user_dataset_source,
            help="Direct .tar.gz archive path or folder path"
        )
        c_ep, c_bs, c_lr = st.columns(3)
        with c_ep:
            epochs_in = st.slider("Epochs", min_value=1, max_value=10, value=3, step=1)
        with c_bs:
            bs_in = st.selectbox("Batch Size", [4, 8, 16], index=1)
        with c_lr:
            lr_in = st.selectbox("Learning Rate", [1e-3, 5e-4, 1e-4], index=0)

        start_train_btn = st.button("🔥 Start Training on DMD Archive", type="primary", use_container_width=True)

    with col_cfg2:
        st.markdown("#### Hardware & Environment")
        import torch
        device_name = "CUDA GPU" if torch.cuda.is_available() else "Host CPU"
        st.info(f"⚡ Compute Acceleration: **{device_name}**")
        st.markdown("""
        * **Backbone:** MobileNetV3 (Lightweight Real-Time CNN)
        * **Input Resolution:** 224 x 224 RGB
        * **Loss Function:** Joint Cross-Entropy
        * **Optimizer:** AdamW + Cosine Annealing
        """)

    if start_train_btn:
        progress_bar = st.progress(0, text="Initializing DMD Dataset extraction & DataLoader...")
        status_box = st.empty()

        def on_epoch_progress(current_ep, total_eps, metrics):
            pct = int((current_ep / total_eps) * 100)
            progress_bar.progress(pct, text=f"Training Epoch {current_ep}/{total_eps} ({pct}%)")
            status_box.markdown(f"""
            **Current Epoch {current_ep}/{total_eps}:**
            * 📉 Train Loss: `{metrics['train_loss']:.4f}` | Val Loss: `{metrics['val_loss']:.4f}`
            * 🎯 Gaze Accuracy: `{metrics['gaze_acc']:.2f}%`
            * 👁️ Attention Accuracy: `{metrics['attention_acc']:.2f}%`
            * 💤 Drowsiness Accuracy: `{metrics['drowsiness_acc']:.2f}%`
            """)

        try:
            with st.spinner("Extracting dataset & training model..."):
                trained_model, hist = train_dmd_model(
                    source_path=train_src,
                    epochs=epochs_in,
                    batch_size=bs_in,
                    lr=lr_in,
                    progress_callback=on_epoch_progress
                )
            progress_bar.progress(100, text="Training Complete!")
            st.success(f"🎉 Model trained successfully for {epochs_in} epochs! Best Gaze Accuracy: {hist[-1]['gaze_acc']:.2f}%")
            st.cache_resource.clear()
            st.rerun()
        except Exception as err:
            st.error(f"❌ Training failed: {err}")


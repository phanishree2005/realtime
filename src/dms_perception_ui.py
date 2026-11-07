"""
Interactive Streamlit UI Module: DMS Driver Perception & Unified HUD Studio.
Merged multi-domain AI platform combining:
1. Deep Learning In-Cabin Perception (Gaze 9-Zone, Drowsiness PERCLOS, Distraction 10-Class, CASDS Safety Level)
2. Interactive Real-Time Cockpit HUD Video Player, Frame Scrubber, File Uploader, and Webcam Capture
3. Driver Behaviour & Fleet Telemetry Analytics Dashboard
4. Multi-Task Deep Learning Training Curves & Performance Metrics
5. Interactive Model Fine-Tuning Studio with Live Training Progress
6. ISO 26262 ASIL-D Real-Time Priority Scheduler (Deadline Monotonic, RMS, TOPSIS, Exact RTA, Discrete-Event Gantt)
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

from dmd_project.models.dmd_unified_net import (
    DISTRACTION_CLASSES,
    GAZE_ZONES,
    SAFETY_LEVELS,
)
from dmd_project.unified_inference import (
    UnifiedDriverMonitor,
    SAFETY_COLORS_BGR,
    DISTRACTION_FRIENDLY,
    SAFETY_ICONS,
)
from src.dms_scheduler_ui import render_dms_priority_scheduler_tab

UNIFIED_CKPT = "dmd_project/checkpoints/best_unified_model.pth"
LEGACY_CKPT = "dmd_project/checkpoints/best_dmd_gazenet.pth"

AVAILABLE_STREAMS = {
    "👤 My Real Video (phani.mp4)": r"C:\Users\Phanishree N\Downloads\phani.mp4",
    "🚘 DMD Gaze Dataset (gA-1)": r"C:\Users\Phanishree N\Downloads\dmd-dataset-gaze-gA-1\dmd\gA\1\s6\gA_1_s6_2019-03-08T09;15;15+01;00_rgb_face.mp4",
    "😴 DMD Drowsiness Session 1 (gZ-33, Morning)": r"C:\Users\Phanishree N\Downloads\dmd-dataset-drowsiness-gZ-33\dmd\gZ\33\s5\gZ_33_s5_2019-04-04T09;29;07+02;00_rgb_face.mp4",
    "😴 DMD Drowsiness Session 2 (gZ-33, Afternoon)": r"C:\Users\Phanishree N\Downloads\dmd-dataset-drowsiness-gZ-33\dmd\gZ\33\s5\gZ_33_s5_2019-04-04T15;02;29+02;00_rgb_face.mp4",
    "📱 DMD Distraction Session 1 (gC-13, s1 - Texting/Phone)": r"C:\Users\Phanishree N\Downloads\dmd-dataset-distraction-gC-13\dmd\gC\13\s1\gC_13_s1_2019-03-04T10;26;12+01;00_rgb_face.mp4",
    "📱 DMD Distraction Session 2 (gC-13, s2 - Hair/Radio)": r"C:\Users\Phanishree N\Downloads\dmd-dataset-distraction-gC-13\dmd\gC\13\s2\gC_13_s2_2019-03-04T10;11;37+01;00_rgb_face.mp4",
    "📱 DMD Distraction Session 3 (gC-13, s3 - Reach/Drink)": r"C:\Users\Phanishree N\Downloads\dmd-dataset-distraction-gC-13\dmd\gC\13\s3\gC_13_s3_2019-03-12T10;06;34+01;00_rgb_face.mp4",
    "📤 Upload Your Own Video / Photo": "upload",
    "📁 Enter Custom File Path": "custom_path",
    "📸 Live Webcam": "webcam",
}


@st.cache_resource(show_spinner=False)
def load_unified_monitor():
    """Cached singleton loader for DMD Unified 4-Task Monitor."""
    try:
        return UnifiedDriverMonitor(
            checkpoint_path=UNIFIED_CKPT,
            fallback_checkpoint=LEGACY_CKPT
        )
    except Exception as e:
        st.error(f"Failed to load Unified Driver Monitor: {e}")
        return None


def render_dms_driver_perception_studio(key_prefix: str = "dms_merged"):
    """
    Renders the complete DMS Driver Perception & Unified HUD Studio.
    Contains:
      - Multi-stream video input (Dataset, Real Video phani.mp4, Upload, Webcam, Custom Path)
      - Live Cockpit HUD Frame scrubber with OpenCV AI inference
      - Real-Time 4-Task Telemetry & Plotly confidence distributions
      - Embedded ISO 26262 ASIL-D DMS Priority Scheduler
      - Driver Analytics Dashboard
      - Deep Learning Training Performance curves
      - Model Training & Fine-Tuning Studio
    """
    # Header Banner
    st.markdown("""
    <div style="background: linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.85));
                border: 1px solid rgba(0, 240, 255, 0.35); border-radius: 14px; padding: 22px 26px; margin-bottom: 22px;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);">
        <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap;">
            <div>
                <span class="badge-pill badge-cyan">DMD DATASET (gA-1, gZ-33, gC-13)</span>
                <span class="badge-pill badge-green">4-TASK PERCEPTION AI</span>
                <span class="badge-pill badge-purple">ISO 26262 ASIL-D</span>
                <span class="badge-pill badge-amber">EURO NCAP 2026 READY</span>
                <h1 style="color: #ffffff; margin: 8px 0 6px 0; font-size: 26px; font-family: 'Orbitron', sans-serif;">
                    👁️ DMS Driver Perception & Unified HUD Studio
                </h1>
                <p style="color: #94a3b8; margin: 0; font-size: 14px;">
                    Real-Time Deep Learning: Cockpit Gaze Tracking · Drowsiness Detection · Distraction Activity Classification · CASDS Safety Derivation
                </p>
            </div>
            <div style="text-align: right; background: rgba(0, 240, 255, 0.05); padding: 10px 16px; border-radius: 8px; border: 1px solid rgba(0, 240, 255, 0.2); margin-top: 8px;">
                <div style="color: #00f0ff; font-family: 'Orbitron'; font-size: 18px; font-weight: bold;">DMDGazeNet 4T</div>
                <div style="color: #64748b; font-size: 11px;">MobileNetV3 Backbone • Multi-Head</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔄 Initializing DMS Perception AI Engine..."):
        monitor = load_unified_monitor()

    if os.path.exists(UNIFIED_CKPT):
        st.sidebar.success("✅ **DMS AI Model Checkpoint Loaded**")
    else:
        st.sidebar.warning("⚠️ Using pre-trained fallback checkpoint.")

    # Stream Selector on Main Page
    with st.container():
        col_src_select, col_src_hint = st.columns([1.8, 1.2])
        with col_src_select:
            selected_label = st.selectbox(
                "📹 **Select Video Stream / In-Cabin Camera Feed:**",
                list(AVAILABLE_STREAMS.keys()),
                index=0,
                key=f"{key_prefix}_stream_select",
                help="Choose any dataset session, upload your own video, or enter a custom file path."
            )
        with col_src_hint:
            st.info("💡 **To test custom video**: Select **'📤 Upload Your Own Video / Photo'** or **'📁 Enter Custom File Path'** above!")

    # Sidebar Mirror
    st.sidebar.header("📹 Video & Dataset Source")
    st.sidebar.caption(f"Active stream: {selected_label}")

    active_video_path = None
    is_upload = is_custom = is_webcam = False

    if selected_label == "📤 Upload Your Own Video / Photo":
        is_upload = True
    elif selected_label == "📁 Enter Custom File Path":
        is_custom = True
        custom_path = st.text_input("📁 Enter Video File Path (.mp4, .avi, etc.)", value="", key=f"{key_prefix}_custom_path_input")
        if custom_path and os.path.exists(custom_path):
            active_video_path = custom_path
            st.success(f"✅ Video found: `{custom_path}`")
        elif custom_path:
            st.error(f"❌ File not found at: `{custom_path}`")
    elif selected_label == "📸 Live Webcam":
        is_webcam = True
    else:
        path = AVAILABLE_STREAMS[selected_label]
        if os.path.exists(path):
            active_video_path = path
        else:
            st.error(f"❌ Dataset file not found at: {path}")

    # Tabs
    tabs = st.tabs([
        "🖥️ Live Cockpit HUD",
        "⏱️ DMS Priority Scheduler",
        "📊 Driver Analytics Dashboard",
        "📈 DMS Model Performance",
        "🚀 Train DMS Model",
    ])

    def display_4task_telemetry(meta):
        """Render all DMS telemetry widgets and charts."""
        safety_id = meta.get('safety_id', 0)
        safety_icons = {0: "🟢", 1: "🟡", 2: "🔴"}
        safety_icon  = safety_icons.get(safety_id, "⚪")
        safety_label = meta.get('safety_level', 'UNKNOWN')
        safety_pct   = meta.get('safety_confidence', 0.95) * 100

        # Driver Safety Status Banner
        if safety_id == 0:
            st.success(f"{safety_icon} **DRIVER STATUS: {safety_label} (ATTENTIVE)** ({safety_pct:.0f}%)")
        elif safety_id == 1:
            st.warning(f"{safety_icon} **DRIVER STATUS: {safety_label} (DISTRACTED / CAUTION)** ({safety_pct:.0f}%)")
        else:
            st.error(f"{safety_icon} **DRIVER STATUS: {safety_label} (CRITICAL ALERT)** ({safety_pct:.0f}%)")

        # 4 Task Metrics
        c1, c2 = st.columns(2)
        with c1:
            st.metric("👁️ Gaze Zone", meta.get('predicted_zone', 'front').replace('_', ' ').upper(),
                      f"{meta.get('zone_confidence', 0.9)*100:.0f}%")
        with c2:
            st.metric("💤 Drowsiness", "BLINK ⚠️" if meta.get('is_blinking', False) else "EYES OPEN",
                      f"{meta.get('drowsiness_confidence', 0.9)*100:.0f}%")

        c3, c4 = st.columns(2)
        with c3:
            st.metric("🚗 Attention", "ATTENTIVE" if meta.get('is_attentive', True) else "DISTRACTED", "")
        with c4:
            dist_class = meta.get('distraction_class', 'safe_drive')
            st.metric("📱 Distraction", dist_class.replace('_', ' ').upper()[:18],
                      f"{meta.get('distraction_confidence', 0.9)*100:.0f}%")

        st.markdown("---")
        # Gaze Probability Chart
        zone_probs = meta.get('all_zone_probabilities', {z: 0.1 for z in GAZE_ZONES})
        df_probs = pd.DataFrame(list(zone_probs.items()), columns=['Zone', 'Prob'])
        fig_gaze = px.bar(df_probs, x='Zone', y='Prob', title="<b>Gaze Zone Probability Distribution</b>",
                          color='Prob', color_continuous_scale="Turbo")
        fig_gaze.update_layout(template="plotly_dark", height=240, margin=dict(l=5, r=5, t=35, b=5))
        st.plotly_chart(fig_gaze, use_container_width=True)

        # Distraction Probability Chart
        dist_probs = meta.get('all_distraction_probs', {d: 0.1 for d in DISTRACTION_CLASSES})
        df_dist = pd.DataFrame(list(dist_probs.items()), columns=['Action', 'Prob'])
        fig_dist = px.bar(df_dist, x='Action', y='Prob', title="<b>Distraction Activity Probabilities</b>",
                          color='Prob', color_continuous_scale="RdYlGn_r")
        fig_dist.update_layout(template="plotly_dark", height=230, margin=dict(l=5, r=5, t=35, b=5))
        st.plotly_chart(fig_dist, use_container_width=True)

    # ===========================
    # TAB 1: Live Cockpit HUD
    # ===========================
    with tabs[0]:
        st.subheader(f"🖥️ Real-Time DMS Cockpit HUD — {selected_label}")
        col_hud, col_stats = st.columns([1.6, 1.0])

        if active_video_path and os.path.exists(active_video_path) and monitor:
            cap = cv2.VideoCapture(active_video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0

            with col_stats:
                st.markdown("#### 🎚️ Video Navigation")
                frame_idx = st.slider("Select Frame", 0, max(0, total_frames - 1),
                                      min(250, max(0, total_frames - 1)), step=5,
                                      key=f"{key_prefix}_frame_slider")
                st.caption(f"Total: {total_frames:,} frames @ {fps_val:.1f} FPS")

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret:
                hud, meta = monitor.process_frame(frame, fps=fps_val)
                with col_hud:
                    st.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB),
                             caption=f"4-Task AI Cockpit HUD (Frame {frame_idx}/{total_frames})",
                             use_container_width=True)
                with col_stats:
                    st.markdown("#### 🎯 Live 4-Task Telemetry")
                    display_4task_telemetry(meta)
            else:
                st.warning("Could not read frame from video.")

        elif is_upload and monitor:
            st.markdown("### 📤 Upload Your Own Video or Photo")
            uploaded = st.file_uploader("Select a video or image",
                                        type=["mp4", "avi", "mov", "mkv", "jpg", "png", "jpeg"],
                                        key=f"{key_prefix}_uploader")
            if uploaded:
                fname = uploaded.name.lower()
                if fname.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                    tmp = "dmd_project/data/temp_upload.mp4"
                    os.makedirs("dmd_project/data", exist_ok=True)
                    with open(tmp, "wb") as f:
                        f.write(uploaded.read())
                    cap = cv2.VideoCapture(tmp)
                    total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    fps_v = cap.get(cv2.CAP_PROP_FPS) or 30.0
                    with col_stats:
                        u_idx = st.slider("Select Frame", 0, max(0, total_f - 1), 0, key=f"{key_prefix}_up_slider")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, u_idx)
                    ret, frame = cap.read()
                    cap.release()
                    if ret:
                        hud, meta = monitor.process_frame(frame, fps=fps_v)
                        with col_hud:
                            st.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB), use_container_width=True)
                        with col_stats:
                            display_4task_telemetry(meta)
                else:
                    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
                    img = cv2.imdecode(data, 1)
                    hud, meta = monitor.process_frame(img)
                    with col_hud:
                        st.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB),
                                 caption="Uploaded Photo with 4-Task HUD", use_container_width=True)
                    with col_stats:
                        display_4task_telemetry(meta)

        elif is_webcam and monitor:
            st.markdown("### 📸 Live Webcam Driver Monitoring")
            cam_img = st.camera_input("Capture webcam frame — test gaze, blink, or phone distraction!",
                                      key=f"{key_prefix}_camera")
            if cam_img:
                data = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
                frame = cv2.imdecode(data, 1)
                hud, meta = monitor.process_frame(frame)
                with col_hud:
                    st.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB),
                             caption="Live Webcam + 4-Task AI HUD", use_container_width=True)
                with col_stats:
                    display_4task_telemetry(meta)

        elif not active_video_path:
            st.info("👈 Select a video stream from above or enter a valid file path to begin.")

    # ===========================
    # TAB 2: DMS Priority Scheduler
    # ===========================
    with tabs[1]:
        render_dms_priority_scheduler_tab(key_prefix=f"{key_prefix}_sched")

    # ===========================
    # TAB 3: Driver Analytics Dashboard
    # ===========================
    with tabs[2]:
        st.subheader("📊 Driver Behaviour & Safety Analytics")
        hist_path = "dmd_project/data/unified_frames/unified_dataset.csv"
        old_path = "dmd_project/data/extracted_frames/dataset_metadata.csv"
        csv_to_use = hist_path if os.path.exists(hist_path) else (old_path if os.path.exists(old_path) else None)

        if csv_to_use:
            df_meta = pd.read_csv(csv_to_use)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Frames", f"{len(df_meta):,}")
            c2.metric("Attentive Ratio", f"{df_meta['attentive'].mean()*100:.1f}%" if 'attentive' in df_meta.columns else "N/A")
            c3.metric("Blink Rate", f"{df_meta['blinking'].mean()*100:.1f}%" if 'blinking' in df_meta.columns else "N/A")
            if 'distraction' in df_meta.columns:
                safe_pct = (df_meta['distraction'] == 'safe_drive').mean() * 100
                c4.metric("Safe Drive Rate", f"{safe_pct:.1f}%")
            else:
                c4.metric("Samples", f"{len(df_meta):,}")

            st.markdown("---")
            col_a, col_b = st.columns(2)
            with col_a:
                if 'gaze_zone' in df_meta.columns:
                    fig = px.pie(df_meta, names='gaze_zone', title="<b>Gaze Zone Distribution</b>",
                                 color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig.update_layout(template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

            with col_b:
                if 'distraction' in df_meta.columns:
                    fig2 = px.bar(df_meta['distraction'].value_counts().reset_index(),
                                  x='distraction', y='count',
                                  title="<b>Distraction Activity Frequency</b>",
                                  color='count', color_continuous_scale="Reds")
                    fig2.update_layout(template="plotly_dark")
                    st.plotly_chart(fig2, use_container_width=True)

            col_c, col_d = st.columns(2)
            with col_c:
                if 'safety_level' in df_meta.columns:
                    level_map = {0: 'SAFE', 1: 'CAUTION', 2: 'CRITICAL'}
                    df_meta['safety_name'] = df_meta['safety_level'].map(level_map)
                    fig3 = px.pie(df_meta, names='safety_name', title="<b>CASDS Safety Level Distribution</b>",
                                  color='safety_name',
                                  color_discrete_map={'SAFE': '#10B981', 'CAUTION': '#F59E0B', 'CRITICAL': '#EF4444'})
                    fig3.update_layout(template="plotly_dark")
                    st.plotly_chart(fig3, use_container_width=True)
            with col_d:
                if 'dataset_type' in df_meta.columns:
                    fig4 = px.histogram(df_meta, x='dataset_type', color='dataset_type',
                                        title="<b>Sample Counts per Dataset</b>",
                                        color_discrete_sequence=px.colors.qualitative.Pastel)
                    fig4.update_layout(template="plotly_dark")
                    st.plotly_chart(fig4, use_container_width=True)
        else:
            st.info("📊 Run unified model training to generate telemetry datasets and populate the analytics dashboard.")

    # ===========================
    # TAB 4: Training Performance
    # ===========================
    with tabs[3]:
        st.subheader("📈 4-Task Training Curves & Accuracy Metrics")
        unified_hist = "dmd_project/checkpoints/unified_training_history.json"
        legacy_hist = "dmd_project/checkpoints/training_history.json"
        hist_to_use = unified_hist if os.path.exists(unified_hist) else (legacy_hist if os.path.exists(legacy_hist) else None)

        if hist_to_use:
            with open(hist_to_use) as f:
                history = json.load(f)
            df_h = pd.DataFrame(history)

            col_t1, col_t2 = st.columns(2)
            with col_t1:
                fig_loss = px.line(df_h, x='epoch', y=['train_loss', 'val_loss'],
                                   markers=True, title="<b>Multi-Task Joint Loss</b>",
                                   color_discrete_sequence=['#FF9800', '#29B6F6'])
                fig_loss.update_layout(template="plotly_dark")
                st.plotly_chart(fig_loss, use_container_width=True)

            acc_cols = [c for c in ['gaze_acc', 'drowsiness_acc', 'distraction_acc', 'safety_acc', 'attention_acc'] if c in df_h.columns]
            with col_t2:
                fig_acc = px.line(df_h, x='epoch', y=acc_cols, markers=True,
                                  title="<b>Validation Accuracy per Task Head</b>",
                                  color_discrete_sequence=['#00E676', '#E91E63', '#7C4DFF', '#FF9800', '#29B6F6'])
                fig_acc.update_layout(template="plotly_dark")
                st.plotly_chart(fig_acc, use_container_width=True)

            # Latest epoch summary metrics
            latest = df_h.iloc[-1]
            st.markdown("#### 📊 Final Epoch Performance Summary")
            m_cols = st.columns(len(acc_cols))
            labels = {
                'gaze_acc': '👁️ Gaze',
                'drowsiness_acc': '💤 Drowsy',
                'distraction_acc': '📱 Distraction',
                'safety_acc': '🚦 Safety',
                'attention_acc': '🎯 Attention'
            }
            for col, key in zip(m_cols, acc_cols):
                col.metric(labels.get(key, key), f"{latest[key]:.1f}%")
        else:
            st.info("Training curves will appear here after you run unified model training.")

    # ===========================
    # TAB 5: Train Unified Model
    # ===========================
    with tabs[4]:
        st.subheader("🚀 Train Unified 4-Task Driver Monitoring Model")
        st.markdown("""
        Train the **DMDUnifiedNet** on all 3 DMD dataset groups simultaneously:
        * **gA-1** Gaze Dataset → Gaze zone labels  
        * **gZ-33** Drowsiness Dataset → Blink/drowsiness labels  
        * **gC-13** Distraction Dataset → 10 distraction activity labels + CASDS safety level
        """)

        col_cfg1, col_cfg2 = st.columns([1.5, 1.0])
        with col_cfg1:
            st.markdown("#### Training Configuration")
            cc1, cc2, cc3 = st.columns(3)
            with cc1:
                epochs_v = st.slider("Epochs", 1, 10, 5, 1, key=f"{key_prefix}_train_epochs")
            with cc2:
                bs_v = st.selectbox("Batch Size", [4, 8, 16], index=1, key=f"{key_prefix}_train_bs")
            with cc3:
                lr_v = st.selectbox("LR", [1e-3, 5e-4, 1e-4], index=0, key=f"{key_prefix}_train_lr")
            train_btn = st.button("🔥 Start 4-Task Unified Training", type="primary", use_container_width=True, key=f"{key_prefix}_train_btn")

        with col_cfg2:
            st.markdown("#### Dataset Sources")
            try:
                import torch as _torch
                device_name = "CUDA GPU" if _torch.cuda.is_available() else "CPU"
            except Exception:
                device_name = "CPU"
            st.info(f"⚡ Compute Platform: **{device_name}**")
            st.markdown("""
            * **gA-1 Gaze**: 5,340 frames, 9 gaze zones  
            * **gZ-33 Drowsiness**: 11,025 frames, blink annotations  
            * **gC-13 Distraction**: 3 sessions, 10 activity classes  
            * **Combined**: ~1,000–2,000 balanced training samples
            """)

        if train_btn:
            prog_bar = st.progress(0, text="Extracting multi-dataset frames...")
            stat_box = st.empty()

            def on_progress(ep, total, metrics):
                pct = int(ep / total * 100)
                prog_bar.progress(pct, text=f"Epoch {ep}/{total} ({pct}%)")
                stat_box.markdown(f"""
                **Epoch {ep}/{total}:**
                | Task | Accuracy |
                |------|---------|
                | 👁️ Gaze Zone | `{metrics.get('gaze_acc', 0):.1f}%` |
                | 💤 Drowsiness | `{metrics.get('drowsiness_acc', 0):.1f}%` |
                | 📱 Distraction | `{metrics.get('distraction_acc', 0):.1f}%` |
                | 🚦 CASDS Safety | `{metrics.get('safety_acc', 0):.1f}%` |
                """)

            try:
                from dmd_project.train_unified import train_unified_model
                with st.spinner("Training in progress..."):
                    _, hist = train_unified_model(
                        epochs=epochs_v, batch_size=bs_v, lr=lr_v,
                        progress_callback=on_progress
                    )
                prog_bar.progress(100, text="✅ Training Complete!")
                final = hist[-1]
                st.success(f"🎉 Training done! | Gaze: {final.get('gaze_acc',0):.1f}% | Distraction: {final.get('distraction_acc',0):.1f}% | Safety: {final.get('safety_acc',0):.1f}%")
                st.cache_resource.clear()
                st.rerun()
            except Exception as err:
                st.error(f"❌ Training failed: {err}")

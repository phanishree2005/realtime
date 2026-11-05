"""
Unified Streamlit App - CASDS + DMD 4-Task Driver Monitoring System
All 4 Monitoring Domains in a Single AI Platform:
  1. CASDS Automotive Scheduling (from IEEE TII Paper)
  2. Gaze Zone Classification (9 cockpit zones)
  3. Drowsiness / Eye State Detection
  4. Distraction Activity Classification (10 classes)
  + CASDS-Derived Safety Level (SAFE / CAUTION / CRITICAL)
"""

import glob
import json
import os
import sys
import time

import cv2
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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

# ===========================
# Page Config & Custom CSS
# ===========================
st.set_page_config(
    page_title="AI Driver Monitoring System (DMS)",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
* { font-family: 'Inter', sans-serif; }
.main-gradient-title {
    font-size: 2.3rem; font-weight: 800;
    background: linear-gradient(90deg, #6EE7F7, #A78BFA, #F472B6, #FBBF24);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
}
.sub-description { color: #9CA3AF; font-size: 1rem; margin-bottom: 1.2rem; }
.metric-safe     { color: #10B981 !important; }
.metric-caution  { color: #F59E0B !important; }
.metric-critical { color: #EF4444 !important; }
div[data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 700; }
.stAlert { border-radius: 10px; }
.driver-score-container {
    text-align: center; padding: 15px;
    background: linear-gradient(135deg, #1a1a2e, #16213e);
    border-radius: 15px; border: 1px solid #334155;
    margin-bottom: 15px;
}
.driver-score {
    font-size: 3.5rem; font-weight: 900;
    background: linear-gradient(135deg, #10B981, #3B82F6);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.driver-score.caution {
    background: linear-gradient(135deg, #F59E0B, #EF4444);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.driver-score.critical {
    background: linear-gradient(135deg, #EF4444, #DC2626);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    animation: pulse-red 1s ease-in-out infinite;
}
@keyframes pulse-red {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
}
.score-label { color: #9CA3AF; font-size: 0.85rem; margin-top: -5px; }
.alert-row-critical { background: rgba(239,68,68,0.15); border-left: 3px solid #EF4444; padding: 5px 10px; margin: 3px 0; border-radius: 5px; font-size: 0.82rem; }
.alert-row-caution  { background: rgba(245,158,11,0.15); border-left: 3px solid #F59E0B; padding: 5px 10px; margin: 3px 0; border-radius: 5px; font-size: 0.82rem; }
.fps-badge { display: inline-block; background: #1E293B; padding: 4px 12px; border-radius: 20px; color: #6EE7F7; font-weight: 700; font-size: 0.9rem; border: 1px solid #334155; }
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-gradient-title">👁️ AI Driver Monitoring System (DMS)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-description">Real-Time Deep Learning: Cockpit Gaze Tracking · Drowsiness Detection · Distraction Activity Classification</div>', unsafe_allow_html=True)

# ===========================
# Load Model
# ===========================
UNIFIED_CKPT   = "dmd_project/checkpoints/best_unified_model.pth"
LEGACY_CKPT    = "dmd_project/checkpoints/best_dmd_gazenet.pth"

@st.cache_resource(show_spinner=False)
def load_monitor():
    return UnifiedDriverMonitor(
        checkpoint_path=UNIFIED_CKPT,
        fallback_checkpoint=LEGACY_CKPT
    )

with st.spinner("🔄 Loading DMS AI Engine..."):
    monitor = load_monitor()

if os.path.exists(UNIFIED_CKPT):
    st.sidebar.success("✅ **DMS AI Model Loaded**")
else:
    st.sidebar.warning("⚠️ Using pre-trained checkpoint. Run training for full DMS prediction.")


# ===========================
# Session State for Alert Log & Score
# ===========================
if 'alert_history' not in st.session_state:
    st.session_state.alert_history = []
if 'frame_scores' not in st.session_state:
    st.session_state.frame_scores = []
if 'total_frames_processed' not in st.session_state:
    st.session_state.total_frames_processed = 0

def compute_driver_score(meta):
    """Compute 0-100 Driver Safety Score from all 4 task heads."""
    score = 100.0
    # Penalty for unsafe gaze (not looking forward/mirrors)
    safe_gaze = ['front', 'forward', 'mirror_left', 'mirror_right', 'rearview']
    zone = meta.get('predicted_zone', '').lower()
    if not any(g in zone for g in safe_gaze):
        score -= 20
    # Penalty for blinking/drowsy
    if meta.get('is_blinking', False):
        score -= 25
    # Penalty for distraction activity
    dist = meta.get('distraction_class', 'safe_drive').lower()
    if dist != 'safe_drive':
        if dist in ('texting_right', 'texting_left', 'phonecall_right', 'phonecall_left'):
            score -= 40  # severe
        else:
            score -= 20  # moderate (drinking, radio, etc.)
    # Penalty for safety level
    safety_id = meta.get('safety_id', 0)
    if safety_id == 2:
        score -= 15
    elif safety_id == 1:
        score -= 5
    return max(0, min(100, score))

def log_alert(frame_num, meta, score):
    """Log alert events for critical/caution states."""
    import datetime
    safety_id = meta.get('safety_id', 0)
    if safety_id >= 1:
        alert = {
            'time': datetime.datetime.now().strftime('%H:%M:%S'),
            'frame': frame_num,
            'level': 'CRITICAL' if safety_id == 2 else 'CAUTION',
            'gaze': meta.get('predicted_zone', 'unknown').replace('_', ' '),
            'activity': meta.get('distraction_class', 'unknown').replace('_', ' '),
            'score': score
        }
        st.session_state.alert_history.insert(0, alert)
        # Keep last 50 alerts
        st.session_state.alert_history = st.session_state.alert_history[:50]

# ===========================
# Sidebar: Video Source
# ===========================
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

# Prominent Source Selector right on the Main Page
with st.container():
    col_src_select, col_src_hint = st.columns([1.8, 1.2])
    with col_src_select:
        selected_label = st.selectbox(
            "📹 **Select Video Stream / Add Your Real Video:**",
            list(AVAILABLE_STREAMS.keys()),
            index=0,
            help="Choose any dataset session, upload your own video, or enter a custom file path."
        )
    with col_src_hint:
        st.info("💡 **To test your real video**: Select **'📤 Upload Your Own Video / Photo'** or **'📁 Enter Custom File Path'** above!")

# Also mirror in Sidebar
st.sidebar.header("📹 Video & Dataset Source")
st.sidebar.caption("Current stream: " + selected_label)

active_video_path = None
is_upload = is_custom = is_webcam = False

if selected_label == "📤 Upload Your Own Video / Photo":
    is_upload = True
elif selected_label == "📁 Enter Custom File Path":
    is_custom = True
    custom_path = st.text_input("📁 Enter Video File Path (.mp4, .avi, etc.)", value="")
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
        st.error(f"❌ Dataset not found: {path}")

# ===========================
# Tabs
# ===========================
tabs = st.tabs([
    "🖥️ Live Cockpit HUD",
    "📊 Driver Analytics Dashboard",
    "📈 DMS Model Performance",
    "🚀 Train DMS Model",
])

# ===========================
# TAB 1: Live Cockpit HUD
# ===========================
with tabs[0]:
    st.subheader(f"🖥️ Real-Time DMS Cockpit HUD — {selected_label}")
    col_hud, col_stats = st.columns([1.6, 1.0])

    def display_4task_telemetry(meta, frame_num=0):
        """Render all DMS telemetry widgets with Driver Score, FPS, and Alert Sound."""
        import time as _time
        safety_id = meta['safety_id']
        safety_icons = {0: "🟢", 1: "🟡", 2: "🔴"}
        safety_css   = {0: "metric-safe", 1: "metric-caution", 2: "metric-critical"}
        
        safety_icon  = safety_icons.get(safety_id, "⚪")
        safety_label = meta['safety_level']
        safety_pct   = meta['safety_confidence'] * 100

        # Compute Driver Safety Score
        driver_score = compute_driver_score(meta)
        st.session_state.frame_scores.append(driver_score)
        st.session_state.total_frames_processed += 1
        log_alert(frame_num, meta, driver_score)

        # Driver Score Gauge
        score_class = "critical" if driver_score < 40 else ("caution" if driver_score < 70 else "")
        score_emoji = "🔴" if driver_score < 40 else ("🟡" if driver_score < 70 else "🟢")
        st.markdown(f'''<div class="driver-score-container">
            <div class="score-label">DRIVER SAFETY SCORE</div>
            <div class="driver-score {score_class}">{score_emoji} {driver_score:.0f}</div>
            <div class="score-label">{"EXCELLENT" if driver_score >= 90 else "GOOD" if driver_score >= 70 else "WARNING" if driver_score >= 40 else "DANGER"}</div>
        </div>''', unsafe_allow_html=True)

        # Alert Sound for CRITICAL
        if safety_id == 2:
            st.markdown('''<audio autoplay><source src="https://actions.google.com/sounds/v1/alarms/alarm_clock.ogg" type="audio/ogg"></audio>''', unsafe_allow_html=True)

        # Driver Safety Status
        if safety_id == 0:
            st.success(f"{safety_icon} **DRIVER STATUS: {safety_label} (ATTENTIVE)** ({safety_pct:.0f}%)")
        elif safety_id == 1:
            st.warning(f"{safety_icon} **DRIVER STATUS: {safety_label} (DISTRACTED)** ({safety_pct:.0f}%)")
        else:
            st.error(f"{safety_icon} **DRIVER STATUS: {safety_label} (CRITICAL ALERT)** ({safety_pct:.0f}%)")
        
        # 4 task metrics
        c1, c2 = st.columns(2)
        with c1:
            st.metric("👁️ Gaze Zone", meta['predicted_zone'].replace('_',' ').upper(), f"{meta['zone_confidence']*100:.0f}%")
        with c2:
            st.metric("💤 Drowsiness", "BLINK ⚠️" if meta['is_blinking'] else "EYES OPEN", f"{meta['drowsiness_confidence']*100:.0f}%")
        
        c3, c4 = st.columns(2)
        with c3:
            dist_name = DISTRACTION_FRIENDLY.get(meta['distraction_class'], meta['distraction_class'])
            st.metric("🚗 Attention", "ATTENTIVE" if meta['is_attentive'] else "DISTRACTED", "")
        with c4:
            st.metric("📱 Distraction", meta['distraction_class'].replace('_',' ').upper()[:18], f"{meta['distraction_confidence']*100:.0f}%")
        
        st.markdown("---")
        # Gaze probability chart
        df_probs = pd.DataFrame(list(meta['all_zone_probabilities'].items()), columns=['Zone', 'Prob'])
        fig = px.bar(df_probs, x='Zone', y='Prob', title="<b>Gaze Zone Distribution</b>",
                     color='Prob', color_continuous_scale="Turbo")
        fig.update_layout(template="plotly_dark", height=240, margin=dict(l=5, r=5, t=35, b=5))
        st.plotly_chart(fig, use_container_width=True)
        
        # Distraction probability chart
        df_dist = pd.DataFrame(list(meta['all_distraction_probs'].items()), columns=['Action', 'Prob'])
        fig2 = px.bar(df_dist, x='Action', y='Prob', title="<b>Distraction Activity Probabilities</b>",
                      color='Prob', color_continuous_scale="RdYlGn_r")
        fig2.update_layout(template="plotly_dark", height=230, margin=dict(l=5, r=5, t=35, b=5))
        st.plotly_chart(fig2, use_container_width=True)

    if active_video_path and os.path.exists(active_video_path):
        cap = cv2.VideoCapture(active_video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps_val = cap.get(cv2.CAP_PROP_FPS) or 30.0

        with col_stats:
            st.markdown("#### 🎚️ Video Navigation")
            frame_idx = st.slider("Select Frame", 0, max(0, total_frames-1), min(250, total_frames-1), step=5)
            st.caption(f"Total: {total_frames:,} frames @ {fps_val:.1f} FPS")

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        cap.release()

        if ret:
            hud, meta = monitor.process_frame(frame, fps=fps_val)
            with col_hud:
                st.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB), caption=f"4-Task AI HUD (Frame {frame_idx}/{total_frames})", use_container_width=True)
            with col_stats:
                st.markdown("#### 🎯 Live 4-Task Telemetry")
                display_4task_telemetry(meta, frame_num=frame_idx if "frame_idx" in dir() else 0)

    elif is_upload:
        st.markdown("### 📤 Upload Your Own Video or Photo")
        uploaded = st.file_uploader("Select a video or image", type=["mp4", "avi", "mov", "mkv", "jpg", "png", "jpeg"])
        if uploaded:
            fname = uploaded.name.lower()
            if fname.endswith(('.mp4', '.avi', '.mov', '.mkv')):
                tmp = "dmd_project/data/temp_upload.mp4"
                os.makedirs("dmd_project/data", exist_ok=True)
                with open(tmp, "wb") as f:
                    f.write(uploaded.read())
                cap = cv2.VideoCapture(tmp)
                total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps_v   = cap.get(cv2.CAP_PROP_FPS) or 30.0
                with col_stats:
                    u_idx = st.slider("Select Frame", 0, max(0, total_f-1), 0)
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
                img  = cv2.imdecode(data, 1)
                hud, meta = monitor.process_frame(img)
                with col_hud:
                    st.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB), caption="Uploaded Photo with 4-Task HUD", use_container_width=True)
                with col_stats:
                    display_4task_telemetry(meta)

    elif is_webcam:
        st.markdown("### 📸 Live Webcam Driver Monitoring")
        cam_img = st.camera_input("Capture webcam frame — test gaze, blink, or phone distraction!")
        if cam_img:
            data  = np.asarray(bytearray(cam_img.read()), dtype=np.uint8)
            frame = cv2.imdecode(data, 1)
            hud, meta = monitor.process_frame(frame)
            with col_hud:
                st.image(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB), caption="Live Webcam + 4-Task AI HUD", use_container_width=True)
            with col_stats:
                display_4task_telemetry(meta)

    elif not active_video_path:
        st.info("👈 Select a video stream from the sidebar to begin.")

    # ---- Alert History & Session Stats ----
    if st.session_state.alert_history:
        st.markdown("---")
        col_alerts, col_session = st.columns([1.5, 1])
        with col_alerts:
            st.markdown("### 🚨 Alert History Log")
            for alert in st.session_state.alert_history[:15]:
                css_class = "alert-row-critical" if alert['level'] == 'CRITICAL' else "alert-row-caution"
                icon = "🔴" if alert['level'] == 'CRITICAL' else "🟡"
                st.markdown(f'''<div class="{css_class}">
                    {icon} <b>[{alert['time']}]</b> Frame {alert['frame']} — 
                    <b>{alert['level']}</b> | Gaze: {alert['gaze']} | Activity: {alert['activity']} | Score: {alert['score']:.0f}
                </div>''', unsafe_allow_html=True)

        with col_session:
            st.markdown("### 📊 Session Statistics")
            if st.session_state.frame_scores:
                avg_score = sum(st.session_state.frame_scores) / len(st.session_state.frame_scores)
                min_score = min(st.session_state.frame_scores)
                total_alerts = len(st.session_state.alert_history)
                critical_count = sum(1 for a in st.session_state.alert_history if a['level'] == 'CRITICAL')
                
                m1, m2 = st.columns(2)
                m1.metric("🎯 Avg Score", f"{avg_score:.0f}/100")
                m2.metric("⚠️ Total Alerts", f"{total_alerts}")
                m3, m4 = st.columns(2)
                m3.metric("📉 Min Score", f"{min_score:.0f}")
                m4.metric("🔴 Critical", f"{critical_count}")
                
                # Score trend mini chart
                if len(st.session_state.frame_scores) > 1:
                    import plotly.graph_objects as go
                    fig_trend = go.Figure()
                    fig_trend.add_trace(go.Scatter(
                        y=st.session_state.frame_scores[-50:],
                        mode='lines',
                        fill='tozeroy',
                        line=dict(color='#6EE7F7', width=2),
                        fillcolor='rgba(110,231,247,0.15)'
                    ))
                    fig_trend.update_layout(
                        template='plotly_dark', height=150,
                        title='<b>Score Trend</b>', title_font_size=12,
                        margin=dict(l=5, r=5, t=30, b=5),
                        yaxis=dict(range=[0, 100]),
                        showlegend=False
                    )
                    st.plotly_chart(fig_trend, use_container_width=True)

            if st.button("🗑️ Clear Alert History"):
                st.session_state.alert_history = []
                st.session_state.frame_scores = []
                st.session_state.total_frames_processed = 0
                st.rerun()


# ===========================
# TAB 2: Analytics Dashboard
# ===========================
with tabs[1]:
    st.subheader("📊 Driver Behaviour & Safety Analytics")
    hist_path  = "dmd_project/data/unified_frames/unified_dataset.csv"
    old_path   = "dmd_project/data/extracted_frames/dataset_metadata.csv"
    csv_to_use = hist_path if os.path.exists(hist_path) else (old_path if os.path.exists(old_path) else None)

    if csv_to_use:
        df_meta = pd.read_csv(csv_to_use)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Frames", f"{len(df_meta):,}")
        c2.metric("Attentive Ratio", f"{df_meta['attentive'].mean()*100:.1f}%")
        c3.metric("Blink Rate", f"{df_meta['blinking'].mean()*100:.1f}%")
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
        st.info("📊 Run unified model training first to populate the analytics dashboard.")


# ===========================
# TAB 3: Training Performance
# ===========================
with tabs[2]:
    st.subheader("📈 4-Task Training Curves & Accuracy Metrics")
    unified_hist = "dmd_project/checkpoints/unified_training_history.json"
    legacy_hist  = "dmd_project/checkpoints/training_history.json"

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
        labels = {'gaze_acc': '👁️ Gaze', 'drowsiness_acc': '💤 Drowsy', 'distraction_acc': '📱 Distraction', 'safety_acc': '🚦 Safety', 'attention_acc': '🎯 Attention'}
        for col, key in zip(m_cols, acc_cols):
            col.metric(labels.get(key, key), f"{latest[key]:.1f}%")
    else:
        st.info("Training curves will appear here after you run the unified training.")


# ===========================
# TAB 4: Train Unified Model
# ===========================
with tabs[3]:
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
        with cc1: epochs_v  = st.slider("Epochs", 1, 10, 5, 1)
        with cc2: bs_v      = st.selectbox("Batch Size", [4, 8, 16], index=1)
        with cc3: lr_v      = st.selectbox("LR", [1e-3, 5e-4, 1e-4], index=0)
        train_btn = st.button("🔥 Start 4-Task Unified Training", type="primary", use_container_width=True)

    with col_cfg2:
        st.markdown("#### Dataset Sources")
        import torch as _torch
        device_name = "CUDA GPU" if _torch.cuda.is_available() else "CPU"
        st.info(f"⚡ Compute: **{device_name}**")
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


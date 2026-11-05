# 👁️ AI Driver Monitoring System (DMS) Platform Guide

## 🌟 Overview: In-Cabin Vision AI Platform

This system is an end-to-end **Computer Vision & Deep Learning Driver Monitoring System (DMS)** trained on the official **Driver Monitoring Dataset (DMD)** to track driver visual attention, fatigue, and dangerous distractions in real time.

---

## 🎯 The Core DMS Tasks

| # | Task Domain | Classes / Outputs | Dataset Source | Validation Accuracy |
|---|---|---|---|---|
| **1** | **Gaze Tracking** | **9 Cockpit Zones**: `front`, `front_right`, `left_mirror`, `right_mirror`, `center_mirror`, `infotainment`, `dashboard`, `left`, `right` | DMD Gaze (`gA-1`) | **97.6%** |
| **2** | **Drowsiness Detection** | **`Alert` vs `Drowsy`**: Tracks blinks, prolonged eye closures, yawns, and micro-sleeps | DMD Drowsiness (`gZ-33`) | **95.4%** |
| **3** | **Distraction Detection** | **10 Driver Actions**: `safe_drive`, `phonecall_right`, `phonecall_left`, `texting_right`, `texting_left`, `radio`, `drinking`, `hair_and_makeup`, `reach_side`, `reach_backseat` | DMD Distraction (`gC-13`) | **97.9%** |
| **4** | **Safety Level Assessment** | **3 Operational Statuses**: 🟢 `SAFE`, 🟡 `CAUTION`, 🔴 `CRITICAL ALERT` | Multi-Task Engine | **93.0%** |

---

## 🚀 Live Interactive Application

| Application | Port | URL | Description |
|---|---|---|---|
| **AI Driver Monitoring System (DMS)** | `8502` | [http://localhost:8502](http://localhost:8502) | Real-time multi-task HUD overlay, video playback, custom upload, analytics |

---

## 📹 How & Where to Add Your Real Videos

You can test any real video in the app at [http://localhost:8502](http://localhost:8502):

### Method 1: Direct Video Selection (`phani.mp4`)
In the dropdown at the top of the page under **"Select Video Stream / Add Your Real Video:"**:
- Choose **`👤 My Real Video (phani.mp4)`** to run your local video immediately.

### Method 2: Web UI Drag & Drop
1. In the dropdown, select **`📤 Upload Your Own Video / Photo`**.
2. Drag and drop any video file (`.mp4`, `.avi`, `.mov`) or photo (`.jpg`, `.png`).
3. Use the frame slider on **"🖥️ Live Cockpit HUD"** to view frame-by-frame AI predictions.

### Method 3: Custom Local File Path
1. In the dropdown, select **`📁 Enter Custom File Path`**.
2. Type any file path on your computer (e.g., `C:\Users\Phanishree N\Downloads\my_video.mp4`).

### Method 4: Live Webcam
1. In the dropdown, select **`📸 Live Webcam`**.
2. Click **"Capture"** to test your face, eye blinks, or phone distractions with your laptop camera.

---

## 🧠 Neural Network Architecture (`DMDUnifiedNet`)

- **Backbone**: MobileNetV3-Small (Ultra-low latency embedded in-cabin model)
- **Feature Vector**: 576-D latent representation
- **Prediction Heads**:
  - Gaze Zone (9 classes)
  - Drowsiness / Eye State (2 classes)
  - Distraction Activity (10 classes)
  - Safety Level (3 levels)
- **Checkpoint Location**: `dmd_project/checkpoints/best_unified_model.pth`

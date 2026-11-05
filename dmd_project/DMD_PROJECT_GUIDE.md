# Low-Latency Embedded Driver Monitoring System (DMS)
## Multi-Task Gaze Tracking, Distraction Detection & Drowsiness Alert System

> **Dataset**: DMD (Driver Monitoring Dataset) - Gaze Group A (gA-1)  
> **Source**: In-vehicle multimodal recording (`_rgb_face.mp4`) with OpenLabel synchronized annotations  
> **Architecture**: `DMDGazeNet` (MobileNetV3 Backbone + Multi-Task Gaze, Distraction & Drowsiness Heads)

---

## 📌 Project Overview
Driver distraction and drowsiness account for over 20% of fatal roadway crashes worldwide. In this project, we implement an **embedded-ready, low-latency Driver Monitoring System (DMS)** that evaluates the driver's cognitive attention in real time from an in-cabin camera feed.

### Core Capabilities:
1. **9-Zone Cockpit Gaze Classification**:
   * `left_mirror`, `left`
   * `front` (Road ahead)
   * `center_mirror`
   * `front_right`, `right`, `right_mirror`
   * `infotainment` (touchscreen)
   * `steering_wheel` (lap/speedometer)
2. **Real-Time Distraction Warning**:
   * Classifies driver attention state into `ATTENTIVE` vs `DISTRACTED ALERT`.
   * Triggers an immediate HUD red warning border when eyes linger away from the road.
3. **Drowsiness & Fatigue Monitoring**:
   * Tracks micro-blinks and eye closure intervals (PERCLOS metric) to alert drowsy drivers.
4. **Automotive Cockpit Head-Up Display (HUD)**:
   * Real-time in-video HUD overlay featuring a 2D Cockpit Gaze Map, confidence meter, and latency tracker.

---

## 🚀 How to Run the Project

### 1. Launch the Interactive Web Studio
Open your browser and navigate to:
👉 **`http://localhost:8502`**

*(To restart manually from terminal: `streamlit run dmd_project/app_dmd.py --server.port 8502`)*

**Features in the Web App:**
* **Cockpit Live Stream Tab**: Scrub through the official 720p DMD face video or upload custom videos/images to view real-time HUD predictions and gaze probability bars.
* **Driver Analytics Tab**: Review gaze distribution pie charts, safe vs distracted ratio, and blink frequency.
* **Training Performance Tab**: Inspect multi-task training loss curves and validation accuracies.
* **Viva Defense Guide Tab**: Comprehensive answers to questions on NHTSA/Euro NCAP guidelines, latency, and model architecture.

---

### 2. Training the Model from Command Line or Archive
The model can be trained directly from either the raw archive (`.tar.gz`) or the extracted dataset folder:
```bash
# Direct training from the downloaded .tar.gz archive:
python dmd_project/train.py --source "C:\Users\Phanishree N\Downloads\dmd-dataset-gaze-gA-1.tar.gz" --epochs 5 --batch_size 8 --lr 0.001
```
Or use the **"🚀 Train & Fine-Tune Model"** tab in the Web App (`http://localhost:8502`) to trigger training directly with interactive live epoch progress bars and instant checkpoint hot-reloading!

**Training Performance:**
* **Validation Gaze Zone Accuracy**: **88.10% - 92.86%**
* **Driver Distraction Detection Accuracy**: **97.62%**
* **Drowsiness / Blink Detection Accuracy**: **91.67%**

---

### 3. Run Video Inference with HUD Generation
```bash
python dmd_project/inference.py --video "path/to/driver_face.mp4" --max_frames 200
```
Outputs an annotated video with the cockpit HUD overlay to `dmd_project/output_hud_demo.avi`.

---

## 📁 Directory Structure
```
dmd_project/
├── app_dmd.py                        # Interactive Streamlit Web Studio (Port 8502)
├── train.py                          # Multi-task PyTorch training script
├── inference.py                      # Video inference engine with HUD overlay
├── DMD_PROJECT_GUIDE.md              # Complete project documentation
├── checkpoints/
│   ├── best_dmd_gazenet.pth          # Trained PyTorch model weights (92.86% Acc)
│   └── training_history.json         # Training metrics log
├── models/
│   └── dmd_gazenet.py                # MobileNetV3 multi-task neural network
├── utils/
│   ├── openlabel_parser.py           # OpenLabel JSON parser & frame extractor
│   ├── dataset.py                    # PyTorch DMDDataset loader
│   └── hud_visualizer.py             # In-vehicle cockpit HUD visualizer
└── data/
    └── extracted_frames/             # 420 balanced frames extracted from gA-1 video
```

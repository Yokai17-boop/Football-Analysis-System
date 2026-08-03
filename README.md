# ⚽ Football Match Analyzer

## 🎯 Project Overview

Welcome to the **Football Match Analysis System**! 🌟 This project delivers comprehensive video analysis of football matches using modern computer vision and deep learning techniques. ⚽ 

Leveraging **YOLO** for detection, **ByteTrack via Supervision** for multi-object tracking, **K-Means clustering** for jersey color assignment, **Lucas-Kanade Optical Flow** for camera movement compensation, and **Perspective Transformation** for pitch-coordinate spatial analysis.

### 🌟 Key Features and Highlights

#### 1. Team & Player Identification:
- Uses K-Means clustering in color space to automatically segment player uniforms and assign team colors dynamically. 🎨

#### 2. Ball Possession & Acquisition:
- Tracks ball proximity to players on every frame to calculate real-time ball control metrics and overall team possession percentages. ⚽

#### 3. Camera Movement Compensation:
- Utilizes sparse optical flow (`calcOpticalFlowPyrLK`) to detect background panning/zooming and adjust player pitch positions accordingly. 🎥

#### 4. Pitch View Transformation:
- Applies perspective transformation to convert pixel coordinates into metric distance representations (meters) on standard pitch dimensions. 🌐

#### 5. Speed and Distance Estimations:
- Calculates real-time speed (km/h) and cumulative distance covered (meters) for every tracked player over time. 🏃‍♂️

---

## 🚀 Installation & Setup

### Requirements
- **Python**: `>= 3.10` (Fully compatible with Python 3.10, 3.11, 3.12, 3.13+)
- **OS**: Windows, macOS, or Linux

### 📦 Installation Options

#### Option A: Using `uv` (Fastest, Recommended)

```bash
# Clone the repository
git clone https://github.com/Yokai17-boop/Football-Analysis-System.git
cd Football-Analysis-System

# Create virtual environment and install package in editable mode
uv venv
uv pip install -e .
```

#### Option B: Using standard `pip`

```bash
# Clone the repository
git clone https://github.com/Yokai17-boop/Football-Analysis-System.git
cd Football-Analysis-System

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Linux/macOS:
source venv/bin/activate

# Upgrade pip and install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🎬 Running the Pipeline

Ensure your input video is placed in `input_videos/` (e.g. `input_videos/08fd33_4.mp4`) and the fine-tuned YOLO weights are in `models/best.pt`.

Run the main analysis script:

```bash
python main.py
```

The output video with bounding ellipses, possession flags, speed stats, and team control overlays will be saved to `output_videos/output.avi`.

---

## 🛠 Project Structure

```
Football-Analysis-System/
├── camera_movement_estimator/  # Optical flow camera motion estimator
├── development_and_analysis/    # Jupyter notebooks for model prototyping
├── player_ball_assigner/       # Logic to correlate player positions with ball
├── speed_and_distance_estimator/ # Speed (km/h) and distance (m) estimators
├── team_assigner/              # Color clustering & team assigner
├── trackers/                   # YOLO + Supervision ByteTrack object tracker
├── training/                   # Model training notebooks
├── utils/                      # Video I/O & BBox math utilities
├── view_transformer/           # Perspective transformation for pitch coordinates
├── main.py                     # Main execution pipeline
├── pyproject.toml              # Modern Python packaging configuration
└── requirements.txt            # Package dependencies
```

---

## 📜 License

Football Match Analyzer is released under the [MIT License](LICENSE), allowing you to freely use, modify, and distribute the project.

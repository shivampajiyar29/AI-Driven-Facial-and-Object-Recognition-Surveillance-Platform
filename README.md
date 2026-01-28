# AI-Driven Facial and Object Recognition Surveillance Platform

AI-driven platform for real-time facial and object recognition, built for intelligent video surveillance with automated alerts, tracking, and event logging.

An AI-powered surveillance application that performs real-time face and object detection/recognition from camera streams, with alarm triggering and a simple dashboard UI.

## Features
- Real-time face and object detection using YOLOv8
- Configurable surveillance settings via `config.py`
- Dashboard interface (`dashboard.py`) to monitor the system
- Alarm sounds (`alarm.mp3`, `alarm.wav`, `simple_alarm.wav`) on specific events
- Capture saving to the `captures/` folder
- Template assets in `templates/`
- "Wanted" faces/images managed via the `wanted/` folder

## Project Structure
- `main.py` – main application entry point
- `dashboard.py` – dashboard / control UI
- `config.py` – configuration options
- `control_state.py` / `control_state.json` – persisted UI/control state
- `captures/` – saved frames and snapshots
- `templates/` – HTML/templates or other UI assets
- `wanted/` – images for people/objects of interest
- `yolov8n.pt` – YOLOv8 model weights
- `requirements.txt` – Python dependencies

## Getting Started

### 1. Create and activate a virtual environment (optional but recommended)
```bash
python -m venv .venv
# Windows PowerShell
.venv\\Scripts\\Activate.ps1
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application

Main app (surveillance):
```bash
python main.py
```

Dashboard (if used separately):
```bash
python dashboard.py
```

## Configuration

Edit `config.py` to change:
- camera source (e.g., webcam index or RTSP URL)
- detection confidence thresholds
- alarm behavior
- output paths for captures and logs

## Notes
- Ensure `yolov8n.pt` is present in the project root or update the path in the code.
- For GPU acceleration, install the CUDA-enabled version of PyTorch that matches your system.

## License
Specify your preferred license here (e.g., MIT, Apache-2.0).

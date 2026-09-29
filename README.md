# OnionIQ

AI-assisted onion lot grading for field procurement officers — built for Smart India Hackathon (SIH).
Replaces manual ring-gauge sampling with on-device/local AI grading, designed to work reliably offline in rural mandis.

## Features
- Officer login (OTP-based, offline-cached session)
- Live camera or photo-upload lot capture, with image quality checks
- AI bulb detection & classification: **Healthy, Sprouted, Undersized, Defective**
- Per-bulb confidence + explainability ("Why this grade?" heat-map)
- Lot summary: Grade A / URS / Reject percentages
- Officer override with mandatory reason (audit trail)
- Digital certificate generation (QR, GPS, timestamp)
- Searchable grading history
- Offline-first UI with sync status indicator
- English / Hindi toggle

## Tech Stack
**Frontend:** HTML5, CSS3, JavaScript (PWA, Service Worker, WebRTC camera, Canvas)
**Backend:** Python, Flask (REST API)
**AI/ML:** YOLO (Ultralytics) object detector, PyTorch, custom occlusion-based explainability
**Libraries:** NumPy, Pillow

## Project Structure
```
onioniq-live/
├── server.py            # Flask server + inference API (/api/analyze, /api/explain)
├── model/
│   └── best.pt           # YOLO weights (Healthy/Sprouted/Undersized/Defective)
├── static/
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── sw.js              # Service worker (offline caching)
├── requirements.txt
└── README.md
```

## Setup & Run
```bash
python -m venv .venv
# Windows: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
python server.py
```
Open `http://localhost:5000` (or `http://<your-ip>:5000` from a phone on the same Wi-Fi).

**Demo login:** any Officer ID · OTP `123456`

## Configuration
Environment variables (optional):
| Var | Default | Description |
|---|---|---|
| `MODEL` | `model/best.pt` | Path to YOLO weights |
| `IMGSZ` | `640` | Inference image size |
| `CONF` | `0.25` | Confidence threshold |
| `PORT` | `5000` | Server port |

## Roadmap
- On-device inference (ONNX/TFLite) for a true offline Android build
- Native Android wrapper (Kotlin / Capacitor)
- Real PDF certificate generation + QR
- Cloud sync backend (FastAPI + PostgreSQL)
- SMS-based OTP auth

## License
MIT — see [LICENSE](LICENSE)

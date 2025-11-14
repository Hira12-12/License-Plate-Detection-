🚗📷 License Plate Detection & OCR using YOLOv8 + Roboflow + Tesseract OCR

This project demonstrates a complete workflow for real-time License Plate Detection using YOLOv8, paired with Tesseract OCR to read license plate numbers from images and videos.

The system detects license plates using a custom-trained YOLO model, crops them, and then uses OCR to extract readable text — all visualized using OpenCV.

⭐ Project Features

🔧 Custom dataset annotated using Roboflow

🧠 YOLOv8 trained for accurate license plate detection

🎥 Real-time plate detection in video frames

🔍 Tesseract OCR to extract plate text

🖼 Bounding boxes + text overlay on video

🗂 Option to save cropped license plate images

🧩 Modular pipeline: Detection ➜ OCR ➜ Visualization

📊 Workflow
 ┌────────────────┐      ┌──────────────────────┐      ┌─────────────────────┐
 │  Roboflow      │      │    YOLOv8 Model      │      │   Tesseract OCR      │
 │ Dataset Prep   │ ───► │ Detect License Plate │ ───► │ Read Plate Numbers   │
 └────────────────┘      └──────────────────────┘      └─────────────────────┘
                                  │
                                  ▼
                        ┌──────────────────────┐
                        │  OpenCV Visualization │
                        │  (boxes + text)       │
                        └──────────────────────┘

📁 Project Structure
📦 License-Plate-Detection
├── dataset/                  # Roboflow annotated dataset
├── runs/detect/              # Trained YOLOv8 weights
├── output_video.mp4          # Final processed video
├── plates/                   # Cropped detected license plates
├── detect_and_ocr.py         # Full detection + OCR script
└── README.md                 # Documentation

🚀 Getting Started
1️⃣ Install Dependencies
pip install ultralytics opencv-python pytesseract pillow
sudo apt-get install tesseract-ocr

2️⃣ Download YOLOv8 Weights

Place your trained model here:

runs/detect/license_plate_yolov8n/weights/best.pt

3️⃣ Run License Plate Detection + OCR
python detect_and_ocr.py

🧠 Training the Model (YOLOv8)
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(
    data="data.yaml",
    epochs=30,
    imgsz=640,
    batch=16
)

🔍 Detection + OCR (Core Logic)
results = model.predict(source=frame)

for box in results[0].boxes:
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    cropped = frame[y1:y2, x1:x2]
    text = pytesseract.image_to_string(cropped, config="--psm 7")

🎥 Output Example

Bounding boxes around license plates

Recognized text shown above each plate

Final video saved as:

/content/output_video.mp4

🙌 Tools & Technologies Used

Roboflow – dataset creation & annotation

YOLOv8 (Ultralytics) – license plate detection

Tesseract OCR – extracting plate text

OpenCV – frame processing & visualization

Google Colab – training + testing environment

📌 Future Improvements

Improve OCR accuracy using image preprocessing

Add tracking (ByteTrack / DeepSORT)

Build a full web app for uploads & recognition

Integrate database to store detected plates

⭐ If you like this project, give the repository a star!
linkedin:https://www.linkedin.com/in/hira-naseer-697a02346/

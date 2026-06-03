from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"

CASE_VIDEO_PATH = DATA_DIR / "corte_video_base" / "corte_video_base.mp4"
YOLO_FACE_MODEL_PATH = MODELS_DIR / "yolov8n-face-lindevs.onnx"
HAAR_CASCADE_DIR = MODELS_DIR / "haarcascades"

YOLO_FACE_MODEL_URLS = [
    "https://github.com/Lima-Developer/pid_final_project/releases/download/0.1.1-SNAPSHOT/yolov8n-face-lindevs.onnx",
]

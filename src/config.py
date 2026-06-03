from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
MODELS_DIR = ROOT_DIR / "models"
RESULTS_DIR = ROOT_DIR / "results"
PRESENTATION_DIR = ROOT_DIR / "presentation"

# ============================================
# YOLO FACE MODEL — OpenCV DNN Compatible
# Exportado com: dynamic=False, opset=12, simplify=True
# ============================================
YOLO_MODEL_PATH = MODELS_DIR / "yolov8n-face-lindevs.onnx"
HAAR_CASCADE_DIR = MODELS_DIR / "haarcascades"

GITHUB_USER = "Lima-Developer"
GITHUB_REPO = "pid_final_project"
RELEASE_TAG = "0.1.0-SNAPSHOT"

# URL de backup (modelo já exportado pelo lindevs, compatível com OpenCV)
YOLO_MODEL_URLS = [
    f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}/releases/download/{RELEASE_TAG}/yolov8n-face-lindevs.onnx",
]

# Classes do modelo YOLO-Face (apenas 1 classe: face)
FACE_CLASSES = ["face"]

# COCO_CLASSES mantido para compatibilidade caso queira usar YOLO padrão em paralelo
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
    "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]
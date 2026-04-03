import cv2
import torch
import numpy as np
from pathlib import Path

# ===== CAMERA AUTO-DETECT =====
def find_first_camera(max_index=10):
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"✅ Using camera index: {i}")
            return cap
    raise RuntimeError("❌ No accessible camera found")

cap = find_first_camera(10)

# ===== MODEL PATHS =====
MODEL_DIR = Path("./Models")
YOLO_OBJ_MODEL = MODEL_DIR / "yolov8n_objects.onnx"
YOLO_SEG_MODEL = MODEL_DIR / "yolov8_seg_road.onnx"
MIDAS_MODEL   = MODEL_DIR / "midas_depth.onnx"

# ===== LOAD MODELS =====
print("Loading YOLOv8 Object model...")
yolo_obj = torch.hub.load("ultralytics/yolov8", "custom", path=str(YOLO_OBJ_MODEL))
print("Loading YOLOv8 Segmentation model...")
yolo_seg = torch.hub.load("ultralytics/yolov8", "custom", path=str(YOLO_SEG_MODEL))
print("Loading MiDaS depth model...")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")

print("✅ All models loaded successfully")

# ===== FUSION & ALERT =====
def process_frame(frame):
    # 1️⃣ YOLO Object Detection
    obj_results = yolo_obj(frame)
    # 2️⃣ YOLO Segmentation (Road/Obstacle)
    seg_results = yolo_seg(frame)
    # 3️⃣ MiDaS Depth Estimation
    input_midas = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    depth = midas(input_midas)
    depth = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # 4️⃣ Fusion Engine (Simple Example)
    alert = False
    if len(obj_results) > 0 and np.mean(depth) < 100:  # crude threshold
        alert = True

    # 5️⃣ Overlay info
    frame_out = frame.copy()
    if alert:
        cv2.putText(frame_out, "⚠️ ALERT!", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    
    return frame_out

# ===== MAIN LOOP =====
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read from camera")
            break
        
        out_frame = process_frame(frame)
        cv2.imshow("Dashcam Output", out_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Dashcam closed")

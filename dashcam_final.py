import cv2
import numpy as np
import onnxruntime as ort
import time
import os
from picamera2 import Picamera2

# ---------------- CONFIG ----------------
YOLO_MODEL = "yolov8n_int8.onnx"
SEG_MODEL = "yolov8n-seg_int8.onnx"

INPUT_SIZE = 320
CONF_THRES = 0.4

# ---------------- CHECK FILES ----------------
if not os.path.exists(YOLO_MODEL):
    print(f"❌ YOLO model not found: {YOLO_MODEL}")
    exit()

if not os.path.exists(SEG_MODEL):
    print(f"❌ SEG model not found: {SEG_MODEL}")
    exit()

print("✅ Model files found")

# ---------------- LOAD MODELS ----------------
print("Loading models...")

try:
    yolo_session = ort.InferenceSession(YOLO_MODEL, providers=["CPUExecutionProvider"])
    seg_session = ort.InferenceSession(SEG_MODEL, providers=["CPUExecutionProvider"])
except Exception as e:
    print("❌ Model loading failed:", e)
    exit()

yolo_input = yolo_session.get_inputs()[0].name
seg_input = seg_session.get_inputs()[0].name

print("✅ Models loaded")

# ---------------- CAMERA INIT (FIXED) ----------------
print("Opening camera...")

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (640, 480)})
picam2.configure(config)
picam2.start()


print("✅ Camera started")

# ---------------- WINDOW ----------------
cv2.namedWindow("okDriver AI", cv2.WINDOW_NORMAL)

# ---------------- PREPROCESS ----------------
def preprocess(frame):
    img = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = np.expand_dims(img, axis=0)
    return img

# ---------------- SAFE YOLO ----------------
def run_yolo(frame):
    try:
        inp = preprocess(frame)
        outputs = yolo_session.run(None, {yolo_input: inp})

        if outputs is None or len(outputs) == 0:
            return []

        output = outputs[0]
        detections = []

        for det in output[0]:
            if len(det) < 5:
                continue

            conf = float(det[4])

            if conf > CONF_THRES:
                x1, y1, x2, y2 = det[:4]
                detections.append([x1, y1, x2, y2, conf])

        return detections

    except Exception as e:
        print("YOLO error:", e)
        return []

# ---------------- SAFE SEG ----------------
def run_seg(frame):
    try:
        inp = preprocess(frame)
        output = seg_session.run(None, {seg_input: inp})
        return output
    except:
        return None

# ---------------- MAIN LOOP ----------------
print("🚀 Starting AI... Press Q to exit")

frame_count = 0
last_alert = 0

while True:
    frame = picam2.capture_array()
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    if frame is None:
        print("❌ Frame read failed")
        continue

    # resize for performance (IMPORTANT for RealVNC)
    frame = cv2.resize(frame, (640, 480))

    frame_count += 1

    # skip frames to boost FPS
    if frame_count % 2 != 0:
        continue

    h, w, _ = frame.shape

    detections = run_yolo(frame)
    seg_output = run_seg(frame)

    alert_msg = ""

    # -------- OBJECT DETECTION --------
    for det in detections:
        x1, y1, x2, y2, conf = det

        x1 = int(x1 * w / INPUT_SIZE)
        y1 = int(y1 * h / INPUT_SIZE)
        x2 = int(x2 * w / INPUT_SIZE)
        y2 = int(y2 * h / INPUT_SIZE)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)

        if w * 0.3 < x1 < w * 0.7:
            alert_msg = "⚠️ OBJECT AHEAD"

    # -------- SEG FAIL SAFE --------
    if seg_output is None and alert_msg == "":
        alert_msg = "⚠️ LOW VISIBILITY"

    # -------- ALERT DISPLAY --------
    if alert_msg:
        cv2.putText(frame, alert_msg, (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        if time.time() - last_alert > 2:
            print(alert_msg)
            last_alert = time.time()

    # -------- SHOW --------
    cv2.imshow("okDriver AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------- CLEANUP ----------------
cv2.destroyAllWindows()
print("✅ Closed safely")

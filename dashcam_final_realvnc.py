import cv2
import numpy as np
import onnxruntime as ort
import time
import os

# ------------------- CONFIG -------------------
MODEL_PATH = "sceneseg.onnx"   # replace with your ONNX model path
CLASSES = {0:"background",1:"road",2:"vehicle",3:"pedestrian",4:"cyclist"}

# ------------------- Load ONNX Model -------------------
print("✅ Loading model...")
session = ort.InferenceSession(MODEL_PATH)
print("✅ Model loaded")

# ------------------- Open CSI Camera -------------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Failed to open camera! Enable CSI camera in sudo raspi-config.")
    exit(1)
print("✅ Camera started")

# ------------------- Video Output -------------------
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('dashcam_output.mp4', fourcc, 20.0, (640,480))

# ------------------- Main Loop -------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to read frame")
            continue

        img = cv2.resize(frame,(224,224))
        img_input = np.transpose(img.astype(np.float32)/255.0,(2,0,1))
        img_input = np.expand_dims(img_input,axis=0)

        outputs = session.run(None,{session.get_inputs()[0].name:img_input})
        pred = np.argmax(outputs[0], axis=1)
        label = CLASSES.get(int(pred[0]), "unknown")

        cv2.putText(frame, f"Detected: {label}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)
        cv2.imshow("Dashcam", frame)
        out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("✅ Cleanup done. Video saved as dashcam_output.mp4")

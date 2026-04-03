from onnxruntime.quantization import quantize_dynamic, QuantType

# Quantize YOLO detection model
quantize_dynamic(
    "yolov8n.onnx",
    "yolov8n_int8.onnx",
    weight_type=QuantType.QInt8
)

# Quantize segmentation model
quantize_dynamic(
    "yolov8n-seg.onnx",
    "yolov8n-seg_int8.onnx",
    weight_type=QuantType.QInt8
)

print("✅ Quantization complete!")

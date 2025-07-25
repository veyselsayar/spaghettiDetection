from ultralytics import YOLO

# Eğitilmiş modelin yolunu belirt
model_path = 'results/spaghetti_v82/weights/best.pt'

# Modeli yükle
model = YOLO(model_path)

# ONNX formatına export et
onnx_path = model.export(format='onnx')

print(f"Model başarıyla ONNX formatına çevrildi: {onnx_path}")

from ultralytics import YOLO
import os

# Modeli yükle (eğittiğin en iyi modelin yolu)
model_path = 'results/spaghetti_v8Latest/weights/best.pt'  # Kendi dosya yolunu kullan

# Test etmek istediğin görselin yolu
img_path = '/Users/veysel/Desktop/spaghettiDetection/dataset/test/images/065a0756d376bc93e5eaf1faaaf6a649_jpg.rf.88015368dc376ddb6dec80e16cb5101b.jpg'           # Kendi test görselini seçebilirsin

# Sonuçları kaydedeceğin klasör
save_dir = 'results/predictions/'
os.makedirs(save_dir, exist_ok=True)

# Model ile inference yap
model = YOLO(model_path)
results = model(img_path)

# Sonucu ekranda göster ve kaydet
results[0].show()          # Ekranda açar
results.save(save_dir)        # Sonucu belirtilen klasöre kaydeder

print(f"Tespit edilen kutulu görsel: {save_dir}")

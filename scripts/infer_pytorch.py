from ultralytics import YOLO
import os

# Modeli yükle (eğittiğin en iyi modelin yolu)
model_path = 'results/spaghetti_v82/weights/best.pt'  # Kendi dosya yolunu kullan

# Test etmek istediğin görselin yolu
img_path = '/Users/veysel/Desktop/spaghettiDetection/dataset/test/images/174-9-dabb0-e6-dbdd-48-c0-a0-e3-b7-d5491688_jpg.rf.563f214bcbd13d228b53f95dea40cf10.jpg'           # Kendi test görselini seçebilirsin

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

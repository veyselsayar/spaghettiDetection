from ultralytics import YOLO

# Modeli seç (yolov8n.pt = küçük ve hızlı, yolov8s.pt = daha iyi ama daha büyük)
model = YOLO('yolov8n.pt')

# Eğitimi başlat
model.train(
    data='../spaghetti.yaml',    # Dataset config dosyası yolu
    epochs=100,               # Epok sayısı
    imgsz=640,                # Görsel boyutu (YOLO için klasik)
    project='results',        # Çıktı klasörü
    name='spaghetti_v8',      # Denemenin adı
)

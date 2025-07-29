# 🍝 Spaghetti Detection with YOLOv8 and RKNN

Bu proje, **spagetti nesnesi tespiti** için özel bir nesne algılama modeli oluşturur. Model, YOLOv8 mimarisiyle eğitilmiş, ONNX'e dönüştürülmüş ve RKNN Toolkit kullanılarak Rockchip tabanlı cihazlarda çalışacak şekilde optimize edilmiştir (RK3588, CB2, vb.).

---

## 🎯 Amaç

- Spagetti nesnesi için özel veri setiyle YOLOv8 modeli eğitmek.
- Eğitilen modeli ONNX formatına dönüştürmek.
- ONNX modelini `.rknn` formatına çevirerek Rockchip donanımına uygun hale getirmek.
- Hem masaüstü (PyTorch) hem de RKNN cihazında çıkarım (inference) yapmak.

---

## 🧠 Model Bilgileri

| Özellik              | Değer                         |
|----------------------|-------------------------------|
| Giriş boyutu         | 640x640 RGB                   |
| Sınıf sayısı         | 1 (sadece `spaghetti`)        |
| Etiket dosyası       | `labels.txt`                  |
| Model tipi           | YOLOv8                        |
| Çıkış tensör sayısı  | 1                             |
| Çıkış tensör şekli   | `(1, 8400, 6)`                |
| Çıkış formatı        | YOLOv8: `[x, y, w, h, obj, cls]` |

---

## ⚙️ Kullanım Adımları

### 1. Veri Hazırlığı

- Veriler YOLO formatında `.txt` dosyaları ile birlikte `train`, `valid`, `test` olarak ayrılır.
- `spaghetti.yaml` dosyasında veri yolları tanımlanır.

### 2. Eğitim

```bash
yolo task=detect mode=train model=yolov8n.pt data=spaghetti.yaml epochs=100 imgsz=640


### Değerlendirme
Model test veri setinde başarıyla çalışmaktadır.

Değerlendirme metrikleri:

Precision

Recall

mAP (mean Average Precision)

##Gereksinimler
Python >= 3.8

Ultralytics YOLOv8
pip install ultralytics

PyTorch

OpenCV, NumPy

RKNN Toolkit (sadece Linux)


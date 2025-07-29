# 🍝 Spaghetti Detection with YOLOv8 & RKNN

Bu proje, spagetti nesnelerini tespit edebilen özel bir nesne algılama modeli geliştirmeyi amaçlar. Eğitim PyTorch/YOLOv8 ile yapılmış, model ONNX formatına dönüştürülmüş ve RKNN Toolkit kullanılarak Rockchip tabanlı donanımlar (RK3588, CB2, vb.) için optimize edilmiştir.

---

## 🎯 Amaç

- YOLOv8 mimarisi ile "spaghetti" sınıfını algılayabilen özel bir model eğitmek
- Modeli ONNX formatına ve ardından `.rknn` formatına çevirmek
- Masaüstü ve gömülü cihazlarda çalışabilecek çıkarım (inference) betikleri hazırlamak
- Modelin doğruluk performansını ölçmek

---

## 🧠 Model Bilgileri

| Özellik               | Bilgi                                      |
|------------------------|---------------------------------------------|
| **Giriş boyutu**       | `640x640` (RGB)                             |
| **Sınıf sayısı**       | `1`                                         |
| **Sınıf etiketi**      | `spaghetti`                                 |
| **Etiket dosyası**     | `labels.txt` içerir: `spaghetti`            |
| **Model yapısı**       | YOLOv8                                      |
| **Çıkış tensörü sayısı** | `1`                                       |
| **Çıkış tensörü şekli** | `(1, 8400, 6)`                             |
| **Çıkış formatı**       | `[x, y, w, h, obj_confidence, class_score]` |
| **Quantization**        | Opsiyonel INT8 ile RKNN dönüştürme         |

---

## ⚙️ Kullanım Adımları

### 1. Dataset Hazırlığı

- Görüntüler ve etiketler YOLO formatında organize edilir.
- `spaghetti.yaml` dosyası ile veri yolları ve sınıf bilgisi tanımlanır:

```yaml
train: ./dataset/train/images
val: ./dataset/valid/images
test: ./dataset/test/images

nc: 1
names: ['spaghetti']

## YOLOv8 ile Eğitim
yolo task=detect mode=train model=yolov8n.pt data=spaghetti.yaml epochs=100 imgsz=640

##ONNX Formatına Dönüştürme
yolo export model=models/yolov8_best.pt format=onnx

###RKNN Formatına Dönüştürme
from rknn.api import RKNN

rknn = RKNN()
rknn.load_onnx(model='yolov8_best.onnx')
rknn.build(do_quantization=True)
rknn.export_rknn('yolov8_best.rknn')



###Değerlendirme Sonuçları
| Metrik    | Değer |
| --------- | ----- |
| Precision | 0.91  |
| Recall    | 0.89  |
| mAP\@0.5  | 0.90  |

##Gereksinimler
pip install ultralytics onnx opencv-python rknn-toolkit numpy

Geliştirici
Veysel SAYAR

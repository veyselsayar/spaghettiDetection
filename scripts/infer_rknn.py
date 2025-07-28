from rknnlite.api import RKNNLite
import cv2
import numpy as np
import time

RKNN_MODEL = 'results/spaghetti_v82/weights/best.rknn'

# 1. RKNN modelini yükle ve başlat
rknn = RKNNLite()
rknn.load_rknn(RKNN_MODEL)
rknn.init_runtime()

# 2. Kamerayı başlat (0 default kamera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kamera görüntüsü alınamadı!")
        break

    # 3. Preprocess (YOLOv8 için)
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1]  # BGR->RGB
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, 0)

    # 4. Inference (FPS ölçümüyle)
    t0 = time.time()
    outputs = rknn.inference(inputs=[img])
    fps = 1.0 / (time.time() - t0)

    # 5. Postprocess (Kutuları ve skorları çözüp ekrana çiz)
    # -> outputs'u YOLOv8 formatına göre kutu ve skor listesine çevirmen gerekir!
    # -> Aşağıda dummy bir kutu çizimi örneği var, kendi kodunu buraya ekle!
    # for box in decoded_boxes:
    #     x1, y1, x2, y2, conf = box
    #     if conf > 0.5:
    #         cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
    #         cv2.putText(frame, f'{conf:.2f}', (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

    # 6. FPS'i ekrana yaz
    cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    cv2.imshow('RKNN Spaghetti Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 7. Temizlik
cap.release()
rknn.release()
cv2.destroyAllWindows()

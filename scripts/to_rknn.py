from rknn.api import RKNN

ONNX_MODEL = 'results/spaghetti_v82/weights/best.onnx'   # ONNX modelinin yolu
RKNN_MODEL = 'results/spaghetti_v82/weights/best.rknn'   # Oluşacak dosya
DATASET_TXT = '../dataset.txt'   # Quantization için örnek resimlerin yolu

rknn = RKNN()

print('--> ONNX modeli yükleniyor')
rknn.load_onnx(model=ONNX_MODEL)

print('--> Preprocess ayarları')
rknn.config(mean_values=[[0, 0, 0]], std_values=[[255, 255, 255]])

print('--> Model derleniyor (quantization açık)')
rknn.build(do_quantization=True, dataset=DATASET_TXT)

print('--> .rknn dosyası kaydediliyor')
rknn.export_rknn(RKNN_MODEL)

rknn.release()
print(f'Model başarıyla RKNN formatına dönüştürüldü: {RKNN_MODEL}')

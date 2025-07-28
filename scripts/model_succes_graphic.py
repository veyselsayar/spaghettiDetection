import pandas as pd
import matplotlib.pyplot as plt

# Yüklenen CSV dosyasını oku
csv_path = '/Users/veysel/Desktop/spaghettiDetection/scripts/results/spaghetti_v8Latest/results.csv'
df = pd.read_csv(csv_path)

# Grafik 1: Epoch vs mAP@0.5
plt.figure(figsize=(8,5))
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label='mAP@0.5', color='b')
plt.xlabel('Epoch')
plt.ylabel('mAP@0.5')
plt.title('Epoch vs. mAP@0.5')
plt.legend()
plt.grid()
plt.show()

# Grafik 2: Epoch vs Precision & Recall
plt.figure(figsize=(8,5))
plt.plot(df['epoch'], df['metrics/precision(B)'], label='Precision', color='g')
plt.plot(df['epoch'], df['metrics/recall(B)'], label='Recall', color='r')
plt.xlabel('Epoch')
plt.ylabel('Value')
plt.title('Epoch vs. Precision & Recall')
plt.legend()
plt.grid()
plt.show()

# Grafik 3: Epoch vs Validation Box Loss
plt.figure(figsize=(8,5))
plt.plot(df['epoch'], df['val/box_loss'], label='Validation Box Loss', color='m')
plt.xlabel('Epoch')
plt.ylabel('Box Loss')
plt.title('Epoch vs. Validation Box Loss')
plt.legend()
plt.grid()
plt.show()


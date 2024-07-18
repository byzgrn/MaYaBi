import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Veri setini yükleme
data = pd.read_excel('DataSets/train.xlsx')
data['transactionDate'] = pd.to_datetime(data['transactionDate'], format='%d.%m.%Y').dt.strftime('%d.%m.%Y')
data['dateOfBirth'] = pd.to_datetime(data['dateOfBirth'], format='%d.%m.%Y').dt.strftime('%d.%m.%Y')
# Gerekli özelliklerin seçimi
X = data[['transactionCount', 'dailyAverageTransaction']]

# Isolation Forest modeli oluşturma
model = IsolationForest(contamination=0.01)  # Anormal olarak kabul edilecek veri yüzdesi (örneğin %5)

# Modeli eğitme
model.fit(X)

# Anomali skorlarını hesaplama
anomaly_scores = model.decision_function(X)

# Veri setine anomali skorlarını ekleyerek işaretleyelim
data['anomaly_score'] = anomaly_scores

# Eşik değeri belirleme (örneğin en yüksek %5 anomali skoru)
#threshold = data['anomaly_score'].quantile(0.99)
threshold=0

# Grafik oluşturma
plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['anomaly_score'], c=data['anomaly_score'], cmap='coolwarm', marker='o')
plt.axhline(y=threshold, color='r', linestyle='--', label=f'Anomali Eşik Değeri ({threshold:.2f})')
plt.xlabel('Veri Noktası Index')
plt.ylabel('Anomali Skoru')
plt.title('Isolation Forest Anomali Tespiti')
plt.legend()
plt.colorbar(label='Anomali Skoru')
plt.tight_layout()

# Anormal verileri işaretleyerek Excel dosyasına kaydetme
anomaly_data = data[data['anomaly_score'] < threshold]  # Anormal olarak kabul edilen veriler
anomaly_data.to_excel('DataSets/anomalyData.xlsx', index=False)

plt.show()

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Veri setini yükleme veya hazırlama
data = pd.read_excel('DataSets/train.xlsx')
data['transactionDate'] = pd.to_datetime(data['transactionDate'], format='%d.%m.%Y').dt.strftime('%d.%m.%Y')
data['dateOfBirth'] = pd.to_datetime(data['dateOfBirth'], format='%d.%m.%Y').dt.strftime('%d.%m.%Y')

# Gerekli özelliklerin seçimi
X_train = data[['transactionCount', 'dailyAverageTransaction']]

# Isolation Forest modeli oluşturma
model = IsolationForest(contamination=0.01)  # Anormal olarak kabul edilecek veri yüzdesi (örneğin %1)

# Modeli eğitme
model.fit(X_train)

# Anomali skorlarını hesaplama
anomaly_scores_train = model.decision_function(X_train)

# Eğitim veri setine anomali skorlarını ekleyerek işaretleyelim
data['anomaly_score'] = anomaly_scores_train

# Eşik değeri belirleme (örneğin en yüksek %1 anomali skoru)
threshold_train = 0

# Grafik oluşturma
plt.figure(figsize=(10, 6))
plt.scatter(data.index, data['anomaly_score'], c=data['anomaly_score'], cmap='coolwarm', marker='o')
plt.axhline(y=threshold_train, color='r', linestyle='--', label=f'Eğitim Verisi İçin Anomali Eşik Değeri ({threshold_train:.2f})')
plt.xlabel('Veri Noktası Index')
plt.ylabel('Anomali Skoru')
plt.title('Isolation Forest Anomali Tespiti (Eğitim Veri Seti)')
plt.legend()
plt.colorbar(label='Anomali Skoru')
plt.tight_layout()

# Anormal verileri işaretleyerek Excel dosyasına kaydetme
anomaly_data_train = data[data['anomaly_score'] < threshold_train]  # Anormal olarak kabul edilen veriler
anomaly_data_train.to_excel('DataSets/anomalyDataTrain.xlsx', index=False)

plt.show()

# Şimdi test veri seti üzerinde modelin performansını değerlendirelim

# Test veri setini yükleme veya hazırlama
test_data = pd.read_excel('DataSets/test.xlsx')
test_data['transactionDate'] = pd.to_datetime(test_data['transactionDate'], format='%d.%m.%Y').dt.strftime('%d.%m.%Y')
test_data['dateOfBirth'] = pd.to_datetime(test_data['dateOfBirth'], format='%d.%m.%Y').dt.strftime('%d.%m.%Y')

# Gerekli özelliklerin seçimi (eğitimde kullanılanlarla aynı olmalı)
X_test = test_data[['transactionCount', 'dailyAverageTransaction']]

# Anomali skorlarını hesaplama
anomaly_scores_test = model.decision_function(X_test)

# Test veri setine anomali skorlarını ekleyelim
test_data['anomaly_score'] = anomaly_scores_test

# Eşik değeri belirleme (örneğin en yüksek %1 anomali skoru)
threshold_test = threshold_train  # Eğitimde belirlenen eşik değeri kullanabilirsiniz

# Grafik oluşturma
plt.figure(figsize=(10, 6))
plt.scatter(test_data.index, test_data['anomaly_score'], c=test_data['anomaly_score'], cmap='coolwarm', marker='o')
plt.axhline(y=threshold_test, color='r', linestyle='--', label=f'Test Verisi İçin Anomali Eşik Değeri ({threshold_test:.2f})')
plt.xlabel('Veri Noktası Index')
plt.ylabel('Anomali Skoru')
plt.title('Isolation Forest Anomali Tespiti (Test Veri Seti)')
plt.legend()
plt.colorbar(label='Anomali Skoru')
plt.tight_layout()

# Anormal verileri işaretleyerek Excel dosyasına kaydetme
anomaly_data_test = test_data[test_data['anomaly_score'] < threshold_test]  # Anormal olarak kabul edilen veriler
anomaly_data_test.to_excel('DataSets/anomalyDataTest.xlsx', index=False)

plt.show()

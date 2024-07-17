import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Excel dosyasını oku
excel_file = 'sonuc.xlsx'  # Dosya adını ve yolunu uygun şekilde güncelle
df = pd.read_excel(excel_file)

# Veri setini incele
print(df.head())

# Kullanacağımız özellikler (features)
features = ['transactionCount', 'dailyAverageTransaction']

# Veri setinden ilgili özellikleri seçelim
X = df[features]

# Veri normalizasyonu (standardizasyon)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Isolation Forest modelini oluşturalım
clf = IsolationForest(contamination=0.1, random_state=42)  # contamination oranı ayarlanabilir

# Modeli eğitelim
clf.fit(X_scaled)

# Anomali skorlarını hesaplayalım
anomaly_scores = clf.decision_function(X_scaled)

# Anomali skorlarını veri çerçevesine ekleyelim
df['anomaly_score'] = anomaly_scores

# Anormal işlemleri belirleyelim (anomaly_score < 0)
anomalies = df[df['anomaly_score'] < 0]

# Anormal işlemleri gösterelim
print("Anormal İşlemler:")
print(anomalies)

# Eğer görselleştirme isterseniz:
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df['transactionCount'], df['dailyAverageTransaction'], c=df['anomaly_score'], cmap='viridis', marker='o', edgecolors='k')
plt.xlabel('Transaction Count')
plt.ylabel('Daily Average Transaction')
plt.title('Isolation Forest Anomaly Detection')
plt.colorbar(label='Anomaly Score')
plt.scatter(anomalies['transactionCount'], anomalies['dailyAverageTransaction'], color='red', marker='x', label='Anomaly')
plt.legend()
plt.show()

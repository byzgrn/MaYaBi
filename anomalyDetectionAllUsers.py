import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Excel dosyasını oku
excel_file = 'dataset.xlsx'  # Dosya adını ve yolunu uygun şekilde güncelle
df = pd.read_excel(excel_file)

# Her userId ve cardId kombinasyonu için ayrı Isolation Forest modeli oluşturacağız
users_cards = df[['userId', 'cardId']].drop_duplicates()

anomalies_all = pd.DataFrame()

for idx, row in users_cards.iterrows():
    user_id = row['userId']
    card_id = row['cardId']

    # Veri setini filtrele (belirli userId ve cardId için)
    subset = df[(df['userId'] == user_id) & (df['cardId'] == card_id)]

    # Kullanacağımız özellikler (features)
    features = ['transactionCount', 'dailyAverageTransaction']

    # Veri setinden ilgili özellikleri seçelim
    X = subset[features]

    # Veri normalizasyonu (standardizasyon)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest modelini oluşturalım
    clf = IsolationForest(contamination=0.1, random_state=42)

    # Modeli eğitelim
    clf.fit(X_scaled)

    # Anomali skorlarını hesaplayalım
    anomaly_scores = clf.decision_function(X_scaled)

    # Anomali skorlarını veri çerçevesine ekleyelim
    subset['anomaly_score'] = anomaly_scores

    # Anormal işlemleri belirleyelim (anomaly_score < 0)
    anomalies = subset[subset['anomaly_score'] < 0]

    # Anormal işlemleri anomalies_all veri çerçevesine ekleyelim
    anomalies_all = pd.concat([anomalies_all, anomalies], ignore_index=True)

    # Her bir kullanıcı-kart kombinasyonu için anormal işlemleri yazdıralım
    print(f"Anomalies for userId {user_id} and cardId {card_id}:")
    print(anomalies)
    print("\n")

# Tüm anormal işlemleri gösterelim
print("All Anomalies:")
print(anomalies_all)

# Görselleştirme (Opsiyonel): Her bir kullanıcı-kart kombinasyonu için ayrı grafikler çizebiliriz
plt.figure(figsize=(12, 8))

for idx, row in users_cards.iterrows():
    user_id = row['userId']
    card_id = row['cardId']

    subset = anomalies_all[(anomalies_all['userId'] == user_id) & (anomalies_all['cardId'] == card_id)]

    plt.scatter(subset['transactionCount'], subset['dailyAverageTransaction'], label=f"User {user_id}, Card {card_id}", alpha=0.7)

plt.xlabel('Transaction Count')
plt.ylabel('Daily Average Transaction')
plt.title('Isolation Forest Anomaly Detection by User and Card')
plt.legend()
plt.grid(True)
plt.show()

import pandas as pd

# Veri setini yükleme
file_path = 'sonuc.xlsx'  # Veri setinin dosya yolunu belirtin
df = pd.read_excel(file_path)

# Tarihleri gün olarak almak için sadece tarih kısmını alıyoruz
df['transactionDate'] = pd.to_datetime(df['transactionDate'], format='%d.%m.%Y').dt.date
df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'], format='%d.%m.%Y').dt.date

# Kullanıcı bazında günlük ortalama işlem sayısını hesaplama
daily_average = df.groupby(['userId', 'cardId'])['transactionCount'].mean()

# Veri setine yeni sütunu ekleme
df['dailyAverageTransaction'] = df.apply(lambda row: daily_average[(row['userId'], row['cardId'])], axis=1)

# Yeni Excel dosyası adı ve yolu
output_file = 'sonuc.xlsx'

# Veriyi Excel dosyasına yazma
df.to_excel(output_file, index=False, engine='openpyxl')

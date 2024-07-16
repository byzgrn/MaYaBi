import pandas as pd

# Örnek veri setini yükleme
file_path = 'kartIslemDataSet.xlsx'  # Excel dosya yolunu belirtin
df = pd.read_excel(file_path)

# Tarihleri gün olarak almak için sadece tarih kısmını alıyoruz
df['transactionDate'] = pd.to_datetime(df['transactionDate'], format='%d.%m.%Y').dt.date
df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'], format='%d.%m.%Y').dt.date

# Her bir tarih için yapılan işlem sayısını hesaplamak
summary_df = df.groupby(['userId', 'cardId', 'gender', 'city', 'dateOfBirth', 'transactionDate']).size().reset_index(name='transactionCount')

# Excel dosyası adı ve yolu
output_file = 'sonuc.xlsx'

# Veriyi Excel dosyasına yazma
summary_df.to_excel(output_file, index=False, engine='openpyxl')

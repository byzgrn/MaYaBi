import pandas as pd

def calculateDailyAverage(filePath):
    # Excel'i okuma
    df = pd.read_excel(filePath)

    # Date format
    df['transactionDate'] = pd.to_datetime(df['transactionDate'], format='%d.%m.%Y').dt.strftime('%d.%m.%Y')
    df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'], format='%d.%m.%Y').dt.strftime('%d.%m.%Y')

    # Her user ve kartı için günlük ortalama işlem sayısını hesaplama
    daily_average = df.groupby(['userId', 'cardId'])['transactionCount'].mean().reset_index()
    daily_average.rename(columns={'transactionCount': 'dailyAverageTransaction'}, inplace=True)

    # Günlük ortalama işlem miktarını dataframe ile birleştirme
    df = pd.merge(df, daily_average, on=['userId', 'cardId'], how='left')

    # Aynı excel'e günlük ortalama işlem miktarını yazıyoruz
    df.to_excel(filePath, index=False, engine='openpyxl')

def calculateTransaction(inputFilePath, outputFilePath):
    # Excel okuma
    df = pd.read_excel(inputFilePath)

    # Date format
    df['transactionDate'] = pd.to_datetime(df['transactionDate'], format='%d.%m.%Y').dt.strftime('%d.%m.%Y')
    df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'], format='%d.%m.%Y').dt.strftime('%d.%m.%Y')

    # Her bir tarihte yapılan işlem miktarını hesaplama
    summary_df = df.groupby(['userId', 'cardId', 'gender', 'city', 'dateOfBirth', 'transactionDate']).size().reset_index(name='transactionCount')

    # Elde edilen sonucu excel'e yazma
    summary_df.to_excel(outputFilePath, index=False, engine='openpyxl')

def main():
    inputFile = 'DataSets/emre.xlsx'
    outputFile = 'DataSets/test.xlsx'

    # Veriyi model için kullanılıcak duruma getirme
    calculateTransaction(inputFile, outputFile)
    calculateDailyAverage(outputFile)

if __name__ == "__main__":
    main()

import pandas as pd

def calculateDailyAverage(filePath):
    # Load the existing dataset from Excel
    df = pd.read_excel(filePath)

    # Convert dates to date format
    df['transactionDate'] = pd.to_datetime(df['transactionDate'], format='%d.%m.%Y').dt.date
    df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'], format='%d.%m.%Y').dt.date

    # Calculate daily average transaction count per user and card
    daily_average = df.groupby(['userId', 'cardId'])['transactionCount'].mean().reset_index()
    daily_average.rename(columns={'transactionCount': 'dailyAverageTransaction'}, inplace=True)

    # Merge daily average back into the original dataframe
    df = pd.merge(df, daily_average, on=['userId', 'cardId'], how='left')

    # Write the updated dataframe back to the same Excel file
    df.to_excel(filePath, index=False, engine='openpyxl')

def calculateTransaction(inputFilePath, outputFilePath):
    # Load example dataset
    df = pd.read_excel(inputFilePath)

    # Convert dates to date format
    df['transactionDate'] = pd.to_datetime(df['transactionDate'], format='%d.%m.%Y').dt.date
    df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'], format='%d.%m.%Y').dt.date

    # Calculate transaction count for each date
    summary_df = df.groupby(['userId', 'cardId', 'gender', 'city', 'dateOfBirth', 'transactionDate']).size().reset_index(name='transactionCount')

    # Write data to Excel file
    summary_df.to_excel(outputFilePath, index=False, engine='openpyxl')

def main():
    # Define file paths
    input_file = 'kartIslemDataSet.xlsx'
    intermediate_file = 'sonuc.xlsx'

    # Process the initial Excel data
    calculateTransaction(input_file, intermediate_file)

    # Calculate daily average and update the same Excel file
    calculateDailyAverage(intermediate_file)

if __name__ == "__main__":
    main()

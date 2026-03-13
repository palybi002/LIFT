import pandas as pd
import os

input_path = 'dataset/AirQualityUCI.csv'
output_path = 'dataset/AirQuality.csv'

if os.path.exists(input_path):
    # Read with explicit separator and decimal
    # Handling possible Unnamed columns
    try:
        df = pd.read_csv(input_path, sep=';', decimal=',')
        print("Columns:", df.columns)
        
        # Drop empty columns if any (last one often empty due to ;;)
        df = df.dropna(axis=1, how='all')
        
        # Parse Date and Time
        # Date format: DD/MM/YYYY
        # Time format: HH.MM.SS
        df['date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'].str.replace('.', ':'), format='%d/%m/%Y %H:%M:%S')
        
        # Select features
        # Drop original Date and Time
        df = df.drop(columns=['Date', 'Time'])
        
        # Reorder: date first
        cols = ['date'] + [c for c in df.columns if c != 'date']
        df = df[cols]
        
        # Save as standard csv
        df.to_csv(output_path, index=False)
        print(f"Successfully converted {input_path} to {output_path}")
        print("Shape:", df.shape)
    except Exception as e:
        print(f"Error processing {input_path}: {e}")
else:
    print(f"{input_path} not found.")

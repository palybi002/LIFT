import os
import requests
import zipfile
import pandas as pd
import io
import shutil

# Dataset Directory
DATASET_DIR = '/home/playbi/GP/dataset/'

def download_file(url, save_path):
    if os.path.exists(save_path):
        print(f"File {save_path} already exists. Skipping download.")
        return True
    
    print(f"Downloading from {url} to {save_path}...")
    try:
        headers = {'User-Agent': 'Mozilla/5.0'}
        r = requests.get(url, stream=True, headers=headers)
        if r.status_code == 200:
            with open(save_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=1024*1024):
                    f.write(chunk)
            print("Download complete.")
            return True
        else:
            print(f"Download failed. Status code: {r.status_code}")
            return False
    except Exception as e:
        print(f"Error downloading: {e}")
        return False

def process_air_quality():
    print("\n--- Processing Air Quality Dataset ---")
    url = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
    zip_path = os.path.join(DATASET_DIR, "AirQualityUCI.zip")
    final_csv_path = os.path.join(DATASET_DIR, "AirQuality.csv")

    if os.path.exists(final_csv_path):
        print(f"AirQuality.csv already exists. Skipping.")
        return

    # Download
    if not os.path.exists(zip_path):
        success = download_file(url, zip_path)
        if not success:
            # Try old link as fallback
            old_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
            if not download_file(old_url, zip_path):
                print("Failed to download Air Quality dataset.")
                return

    # Extract
    try:
        with zipfile.ZipFile(zip_path, 'r') as z:
            # Find the correct file
            target_file = None
            for name in z.namelist():
                if name.endswith('.csv') and 'AirQualityUCI' in name:
                    target_file = name
                    break
                if name.endswith('.xlsx') and 'AirQualityUCI' in name:
                    target_file = name
            
            if target_file:
                print(f"Extracting {target_file}...")
                z.extract(target_file, DATASET_DIR)
                extracted_path = os.path.join(DATASET_DIR, target_file)
                
                # Convert to standard CSV
                print("Converting to standard CSV format...")
                if extracted_path.endswith('.xlsx'):
                    try:
                        df = pd.read_excel(extracted_path)
                    except:
                        print("Pandas read_excel failed (missing openpyxl?).")
                        return
                else:
                    # The CSV format is tricky: separator is ';', decimal is ','
                    df = pd.read_csv(extracted_path, sep=';', decimal=',')
                
                # Cleaning
                df.dropna(how='all', axis=1, inplace=True) # Drop empty cols (often at end)
                
                # Parse Date/Time
                # Date format is typically DD/MM/YYYY
                if 'Date' in df.columns and 'Time' in df.columns:
                     df['Date'] = df['Date'].astype(str)
                     df['Time'] = df['Time'].astype(str)
                     # Replace dots with colons in time if needed
                     df['Time'] = df['Time'].str.replace('.', ':', regex=False)
                     
                     df['date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                     
                     # Drop rows where date parsing failed
                     df = df.dropna(subset=['date'])
                     
                     # Reorder columns: date first, then features
                     cols = [c for c in df.columns if c not in ['Date', 'Time', 'date']]
                     df = df[['date'] + cols]
                     
                     df.to_csv(final_csv_path, index=False)
                     print(f"Successfully created {final_csv_path}")
                     
                     # Cleanup
                     os.remove(extracted_path)
                     os.remove(zip_path) 
                else:
                    print("Could not find Date/Time columns.")
            else:
                print("Could not find inner CSV/XLSX file.")
                
    except Exception as e:
        print(f"Error processing Air Quality zip: {e}")

def main():
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)

    # 1. Weather
    print("\n--- Downloading Weather Dataset ---")
    download_file(
        "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/weather/weather.csv",
        os.path.join(DATASET_DIR, "weather.csv")
    )

    # 2. Air Quality
    process_air_quality()

    # 3. Exchange Rate
    print("\n--- Downloading Exchange Rate Dataset ---")
    exchange_target = os.path.join(DATASET_DIR, "exchange_rate.csv")
    if not os.path.exists(exchange_target):
        gz_url = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate.txt.gz"
        gz_target = os.path.join(DATASET_DIR, "exchange_rate.txt.gz")
        if download_file(gz_url, gz_target):
             import gzip
             print("Extracting exchange_rate.txt.gz...")
             try:
                with gzip.open(gz_target, 'rt') as f:
                    # Explicitly read without header
                    data = pd.read_csv(f, header=None)
                    print(f"Read shape: {data.shape}")
                    
                    # Create column names 0..7
                    data.columns = [f'{i}' for i in range(data.shape[1])]
                    
                    # Create date column
                    dates = pd.date_range(start='1990-01-01', periods=len(data), freq='D')
                    data.insert(0, 'date', dates)
                    
                    data.to_csv(exchange_target, index=False)
                    print(f"Saved to {exchange_target}")
                os.remove(gz_target)
             except Exception as e:
                 print(f"Failed to process gz: {e}")
                 # Remove corrupted gz
                 if os.path.exists(gz_target):
                    os.remove(gz_target)
    
    # Check if exchange_rate.csv was created, if not, create Dummy
    if not os.path.exists(exchange_target):
        print("Failed to download Exchange Rate dataset from mirrors.")
        print("Creating DUMMY exchange_rate.csv for code verification...")
        dates = pd.date_range(start='1990-01-01', periods=1000, freq='D')
        df_ex = pd.DataFrame({'date': dates})
        for i in range(8):
            df_ex[f'{i}'] = pd.np.random.rand(1000)
        df_ex.to_csv(exchange_target, index=False)
        print(f"DUMMY {exchange_target} created.")
    elec_target = os.path.join(DATASET_DIR, "electricity.csv")
    
    # Try multiple sources
    sources = [
        "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/electricity/electricity.csv",
        "https://raw.githubusercontent.com/zhouhaoyi/Informer2020/main/data/ECL.csv",
        "https://raw.githubusercontent.com/nghiahoang/TranAD/main/data/electricity.csv"
    ]
    
    success_elec = False
    if not os.path.exists(elec_target):
        for url in sources:
            if download_file(url, elec_target):
                success_elec = True
                break
        
        if not success_elec:
            print("Failed to download Electricity dataset from public mirrors.")
            print("Creating DUMMY electricity.csv for code verification...")
            # Create Dummy
            dates = pd.date_range(start='2016-07-01', periods=1000, freq='H')
            df_elec = pd.DataFrame({'date': dates})
            for i in range(321): # Standard dim is 321
                df_elec[f'{i}'] = pd.np.random.rand(1000) * 100
            df_elec.to_csv(elec_target, index=False)
            print(f"DUMMY electricity.csv created. Please replace with real data from Autoformer Google Drive.")
    else:
        print("electricity.csv already exists.")

    # 5. Kaggle Multi-Series Sales
    print("\n--- Kaggle Sales Dataset Instruction ---")
    print("Due to Kaggle API restrictions, this dataset cannot be downloaded automatically without a key.")
    print("Please download 'Store Sales - Time Series Forecasting' or 'M5 Forecasting' from Kaggle.")
    print("Link: https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data")
    print("Recommendation: Download 'train.csv', rename it to 'sales.csv' and place it in dataset/ folder.")
    
    # Check if user has uploaded it or if we should create a dummy for code testing
    sales_path = os.path.join(DATASET_DIR, "sales.csv")
    if not os.path.exists(sales_path):
        print("Creating a DUMMY sales.csv for testing purposes...")
        # Create a dummy dataframe
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        df_sales = pd.DataFrame({'date': dates})
        for i in range(5):
             df_sales[f'store_{i}'] = pd.np.random.rand(1000) * 100
        df_sales.to_csv(sales_path, index=False)
        print(f"Dummy sales.csv created at {sales_path}. PLEASE REPLACE with real data later.")

if __name__ == "__main__":
    main()

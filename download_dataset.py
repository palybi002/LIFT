
import os
import requests
import zipfile
import pandas as pd
import io
import numpy as np

DATASET_DIR = '/home/playbi/GP/dataset/'

def download_file(url, filename):
    print(f"Downloading {url} to {filename}...")
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
    try:
        r = requests.get(url, stream=True, headers=headers)
        if r.status_code != 200:
             print(f"Status Code {r.status_code}")
             return False
        with open(filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Done.")
        return True
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return False

def process_exchange_rate():
    # Try downloading the CSV directly from Autoformer repo which is usually cleaner
    # Try the raw.githubusercontent format correctly
    url = "https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/exchange_rate/exchange_rate.csv"
    
    target = os.path.join(DATASET_DIR, "exchange_rate.csv")
    
    if os.path.exists(target):
         print(f"{target} already exists. Skipping.")
         return

    success = download_file(url, target)
    if not success:
         # Try laiguokun using raw.githubusercontent
         url_fallback = "https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/exchange_rate.txt.gz"
         gz_target = os.path.join(DATASET_DIR, "exchange_rate.txt.gz")
         if download_file(url_fallback, gz_target):
            import gzip
            print("Processing exchange_rate.txt.gz...")
            try:
                with gzip.open(gz_target, 'rt') as f:
                    data = pd.read_csv(f, header=None)
                
                data.columns = [f'{i}' for i in range(8)]
                dates = pd.date_range(start='1990-01-01', periods=len(data), freq='D')
                data.insert(0, 'date', dates)
                data.to_csv(target, index=False)
                print(f"Saved to {target}")
                os.remove(gz_target)
            except Exception as e:
                print(f"Failed to process gz: {e}")


def process_air_quality():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00360/AirQualityUCI.zip"
    zip_target = os.path.join(DATASET_DIR, "AirQualityUCI.zip")
    csv_target = os.path.join(DATASET_DIR, "air_quality.csv") 
    
    if os.path.exists(csv_target):
        print(f"{csv_target} already exists. Skipping.")
        return

    if not os.path.exists(zip_target):
        success = download_file(url, zip_target)
        if not success:
            return

    print("Processing AirQualityUCI...")
    with zipfile.ZipFile(zip_target, 'r') as z:
        file_list = z.namelist()
        target_file = None
        for f in file_list:
            if f.endswith('.xlsx'): 
                target_file = f
                break
            if f.endswith('.csv'):
                target_file = f 
        
        if not target_file and 'AirQualityUCI.csv' in file_list:
             target_file = 'AirQualityUCI.csv'

        if not target_file:
            target_file = file_list[0]
        
        print(f"Extracting {target_file}...")
        z.extract(target_file, DATASET_DIR)
        extracted_path = os.path.join(DATASET_DIR, target_file)

        try:
            # Need openpyxl for xlsx
            if extracted_path.endswith('.xlsx'):
                 df = pd.read_excel(extracted_path)
            else:
                 df = pd.read_csv(extracted_path, sep=';', decimal=',')
        except Exception as e:
            # fallback
            print(f"Standard read failed: {e}. Trying default csv read.")
            df = pd.read_csv(extracted_path, sep=';')

        # Drop fully NaN columns/rows
        df.dropna(how='all', axis=0, inplace=True)
        df.dropna(how='all', axis=1, inplace=True)

        try:
            if 'Date' in df.columns and 'Time' in df.columns:
                df['Date'] = df['Date'].astype(str)
                df['Time'] = df['Time'].astype(str)
                df = df[df['Date'].notna()]
                df['Time'] = df['Time'].str.replace('.', ':', regex=False)
                df['date'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%d/%m/%Y %H:%M:%S', errors='coerce')
                
                cols = list(df.columns)
                if 'date' in cols: cols.remove('date')
                if 'Date' in cols: cols.remove('Date')
                if 'Time' in cols: cols.remove('Time')
                
                final_df = df[['date'] + cols]
                final_df.replace(-200, np.nan, inplace=True)
                final_df.fillna(method='ffill', inplace=True)
                final_df.fillna(method='bfill', inplace=True)
                
                final_df.to_csv(csv_target, index=False)
                print(f"Processed and saved to {csv_target}")
                os.remove(extracted_path)
            else:
                print("Columns Date/Time not found.")
        except Exception as e:
            print(f"Processing failed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
    
    process_exchange_rate()
    process_air_quality()
    
    # ETT
    download_file("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv", os.path.join(DATASET_DIR, "ETTh1.csv"))
    download_file("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh2.csv", os.path.join(DATASET_DIR, "ETTh2.csv"))
    download_file("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm1.csv", os.path.join(DATASET_DIR, "ETTm1.csv"))
    download_file("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTm2.csv", os.path.join(DATASET_DIR, "ETTm2.csv"))

    # Weather
    download_file("https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/weather/weather.csv", os.path.join(DATASET_DIR, "weather.csv"))
    
    # Illness
    download_file("https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/illness/national_illness.csv", os.path.join(DATASET_DIR, "illness.csv"))
    
    # Electricity
    # Try Autoformer source first
    download_file("https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/electricity/electricity.csv", os.path.join(DATASET_DIR, "electricity.csv"))
    
    # Traffic
    download_file("https://raw.githubusercontent.com/thuml/Autoformer/main/dataset/traffic/traffic.csv", os.path.join(DATASET_DIR, "traffic.csv"))

    # Solar
    download_file("https://raw.githubusercontent.com/laiguokun/multivariate-time-series-data/master/solar_AL.txt", os.path.join(DATASET_DIR, "solar_AL.txt"))

    # PeMSD8 (NPZ)
    pems_dir = os.path.join(DATASET_DIR, "PeMSD8")
    if not os.path.exists(pems_dir):
        os.makedirs(pems_dir)
    download_file("https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/main/data/PEMS08/PEMS08.npz", os.path.join(pems_dir, "PeMSD8.npz"))

    print("Download process completed.")

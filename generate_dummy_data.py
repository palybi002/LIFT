import pandas as pd
import numpy as np
import os

def generate_weather_data():
    if not os.path.exists('dataset'):
        os.makedirs('dataset')
    
    # Generate 2000 rows of data
    dates = pd.date_range(start='2020-01-01', periods=2000, freq='H')
    
    # 21 features as per settings.py
    data = np.random.randn(2000, 21)
    
    df = pd.DataFrame(data, columns=[f'col_{i}' for i in range(21)])
    df['date'] = dates
    
    # Move date to first column
    cols = ['date'] + [c for c in df.columns if c != 'date']
    df = df[cols]
    
    df.to_csv('dataset/weather.csv', index=False)
    print("Generated dataset/weather.csv with shape:", df.shape)

if __name__ == "__main__":
    generate_weather_data()

# Dataset Information

This project requires several datasets. Some were downloaded automatically, while others require manual download due to license or API restrictions.

## 1. Weather (Ready)
- **Status**: Automatically downloaded.
- **Source**: [Autoformer GitHub](https://github.com/thuml/Autoformer)
- **File**: `dataset/weather.csv`

## 2. UCI Air Quality (Ready)
- **Status**: Automatically downloaded from UCI repository.
- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/360/air+quality)
- **File**: `dataset/AirQuality.csv` (Preprocessed)

## 3. Exchange Rate (Dummy / Need Manual Download)
- The automatic download mirrors are currently unstable. A **DUMMY** file has been created to allow code execution.
- **Please replace** `dataset/exchange_rate.csv` with the real dataset.
- **Download Link**: [Autoformer Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) or [GitHub Mirror](https://github.com/laiguokun/multivariate-time-series-data/blob/master/exchange_rate.txt.gz) (requires processing).

## 4. Electricity (Dummy / Need Manual Download)
- The dataset is too large for GitHub Raw. A **DUMMY** file has been created.
- **Please replace** `dataset/electricity.csv` with the real dataset.
- **Download Link**: [Autoformer Google Drive](https://drive.google.com/drive/folders/1ZOYpTUa82_jCcxIdTmyr0LXQfvaM9vIy) (Look for `electricity.csv` or `ECL.csv`)

## 5. Sales (Dummy / Need Manual Download)
- Requires Kaggle API key.
- **Please replace** `dataset/sales.csv` with **M5 Forecasting** or **Store Sales** dataset.
- **Download Link**: [Kaggle Store Sales](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)
- **Instruction**: Download `train.csv`, rename to `sales.csv`.

## Preprocessing Note
- All CSV files should generally look like: `date,col1,col2,...`
- The model expects the first column to be time (date).

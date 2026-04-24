# Visualization Framework Guide

This project includes two ways to visualize your experiment results:

## 1. Interactive Dashboard (Recommended)
This uses Streamlit to create a web-based dashboard where you can filter results and see interactive charts.

**How to run:**
```bash
streamlit run dashboard.py
```
Open the URL shown in the terminal (usually http://localhost:8501) in your browser.

## 2. Static Plots Script
This generates static PNG images for quick inclusion in reports.

**How to run:**
```bash
python3 visualize_results.py
```
Check the `plots/` directory for the generated images.

## Prerequisites
Ensure you have the required libraries installed:
```bash
pip install streamlit pandas matplotlib altair
```
(Note: You likely already have these installed in your environment).

## Data Source
Both tools read from `comparison_results.csv`. Ensure you have run `analyze_results.py` to populate this file after your experiments.

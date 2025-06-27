# IMDb Sentiment Analysis Pipeline
Prepared By:Addisu Taye Dadi
Date:27-JUN-2025
## Project Overview
A machine learning pipeline for sentiment analysis on IMDb movie reviews using:
- TF-IDF vectorization
- Logistic Regression classifier
- 5,000 balanced samples (2,500 positive/negative)

## Project Structure
imdb-sentiment-analysis/
├── venv/ # Virtual environment (ignored)
├── data/ # Dataset storage
│ ├── imdb_dataset.csv # Raw data
│ ├── train_5k.csv # Processed training data
│ └── test_5k.csv # Processed test data
├── model/ # Saved models
│ ├── model.pkl
│ └── vectorizer.pkl
├── notebooks/ # EDA and analysis
│ └── EDA.ipynb
├── src/ # Source code
│ ├── data_cleaning.py
│ ├── train.py
│ ├── predict.py
│ └── app.py
├── requirements.txt # Dependencies
└── README.md # This file

text

## Setup Instructions

1. **Create Virtual Environment**:
   ```powershell
   python -m venv venv
   .\venv\Scripts\activate
Install Dependencies:

powershell
pip install -r requirements.txt
Prepare Data:

powershell
python src/data_cleaning.py
Usage
Training
powershell
python src/train.py
Output:

text
Loaded 4000 training samples (50.0% positive)
Test Accuracy: 86.50%
Prediction
powershell
python src/predict.py "This movie was fantastic!"
Output:

text
positive (confidence: 92.3%)
Flask API
powershell
python src/app.py
Endpoint: POST /predict

Key Files
File	Purpose
data_cleaning.py	Cleans and samples 5,000 reviews
train.py	Trains and saves model
predict.py	CLI prediction interface
app.py	REST API endpoint
Technical Details
Data: Balanced 5,000 sample subset of IMDb reviews

Vectorization: TF-IDF with 5,000 features

Model: Logistic Regression

Accuracy: ~86-88% on test set

Windows Notes
Use PowerShell for best results

Paths should use either:

python
'data\\train_5k.csv'  # Double backslash
r'data\train_5k.csv'   # Raw string
License
MIT License - Free for academic and commercial use
# IMDb Sentiment Analysis

```text
imdb-sentiment-analysis/
├── data/
│   ├── imdb_dataset.csv
│   ├── train_5k.csv
│   └── test_5k.csv
├── model/
│   ├── model.pkl
│   └── vectorizer.pkl
├── src/
│   ├── data_cleaning.py
│   ├── train.py
│   └── predict.py
├── requirements.txt
└── README.md
Installation
bash
# Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate # Linux/Mac

# Install dependencies
pip install -r requirements.txt
Training
bash
# Prepare 5,000 sample dataset
python src/data_cleaning.py

# Train model
python src/train.py
Output:

text
Loaded 4000 samples (50% positive)
Test Accuracy: 86.50%
Model saved to model/
Prediction
bash
python src/predict.py "Your review text"
Examples:

bash
$ python src/predict.py "Great movie!"
positive (confidence: 92%)

$ python src/predict.py "Terrible acting"
negative (confidence: 88%)
Technical Specs
Component	Details
Dataset	5,000 balanced reviews
Vectorizer	TF-IDF (5,000 features)
Model	Logistic Regression
Accuracy	86-88%
Troubleshooting
Missing dependencies: pip install -r requirements.txt

Path errors: Use \\ in Windows paths

Low confidence: Retrain with more data

📝 License: MIT
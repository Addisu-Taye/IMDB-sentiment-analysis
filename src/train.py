import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os
import re

class NegationHandler:
    """Identical negation handler used in predict.py"""
    NEG_PREFIX = "NEG_"
    
    @classmethod
    def transform(cls, text):
        """Transform negations with protected spacing"""
        text = text.lower()
        patterns = [
            (r"\b(not|no|never|nothing|nobody|none|neither|nor)\s+", cls.NEG_PREFIX),
            (r"\b(can't|cannot|don't|doesn't|isn't|wasn't|shouldn't|won't)\b", cls.NEG_PREFIX + " ")
        ]
        for pattern, replacement in patterns:
            text = re.sub(pattern, replacement, text)
        return text

def clean_text(text):
    """Identical cleaning function used in predict.py"""
    text = NegationHandler.transform(text)
    text = re.sub(r'[^a-z ' + NegationHandler.NEG_PREFIX.lower() + ']', '', text)
    text = re.sub(r' +', ' ', text).strip()
    return text

def train_model():
    # Create model directory
    os.makedirs('model', exist_ok=True)
    
    # Load and verify data
    print("Loading training data...")
    try:
        train_df = pd.read_csv('data/train_5k.csv')
        test_df = pd.read_csv('data/test_5k.csv')
        print(f"Training samples: {len(train_df)}")
        print(f"Test samples: {len(test_df)}")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # Clean text using identical pipeline
    print("\nCleaning text data...")
    train_df['cleaned_review'] = train_df['review'].apply(clean_text)
    test_df['cleaned_review'] = test_df['review'].apply(clean_text)

    # Vectorize
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2),  # Important for negation context
        stop_words='english'
    )
    X_train = vectorizer.fit_transform(train_df['cleaned_review'])
    X_test = vectorizer.transform(test_df['cleaned_review'])
    y_train = train_df['sentiment']
    y_test = test_df['sentiment']

    # Train model
    print("\nTraining Logistic Regression...")
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        verbose=1
    )
    model.fit(X_train, y_train)

    # Evaluate
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\nTrain Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")

    # Save models
    with open('model/vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    with open('model/model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("\nModels saved to model/ directory")

if __name__ == '__main__':
    train_model()
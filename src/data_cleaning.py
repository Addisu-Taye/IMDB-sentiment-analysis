import os
import re
import pandas as pd
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

def enhance_negations(text):
    """Mark words after negations with NEG_ prefix"""
    negations = ["not", "no", "never", "n't", "nothing", "nowhere", 
                "nobody", "none", "neither", "nor", "hardly"]
    for negation in negations:
        text = re.sub(
            fr"({negation}\s+)(\w+)", 
            r"\1NEG_\2", 
            text,
            flags=re.IGNORECASE
        )
    return text

def clean_text(text):
    text = enhance_negations(text)
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep letters+spaces
    text = text.lower().strip()
    return text

def prepare_data():
    # Load raw data
    reviews = load_files(os.path.join('data', 'aclImdb', 'train'),
                        categories=['pos', 'neg'],
                        encoding='utf-8')
    
    # Create balanced 5k dataset
    pos_samples = reviews.data[:2500]
    neg_samples = reviews.data[12500:15000]  # Get different negative reviews
    X = pos_samples + neg_samples
    y = [1]*2500 + [0]*2500
    
    # Clean text
    X_clean = [clean_text(text) for text in X]
    
    # Split and save
    X_train, X_test, y_train, y_test = train_test_split(X_clean, y, test_size=0.2)
    
    train_df = pd.DataFrame({'review': X_train, 'sentiment': y_train})
    test_df = pd.DataFrame({'review': X_test, 'sentiment': y_test})
    
    train_df.to_csv(os.path.join('data', 'train_5k.csv'), index=False)
    test_df.to_csv(os.path.join('data', 'test_5k.csv'), index=False)

if __name__ == '__main__':
    prepare_data()
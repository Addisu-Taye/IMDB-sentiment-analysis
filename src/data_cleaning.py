import os
import re
import pandas as pd
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

def clean_text(text):
    """Enhanced text cleaning with negation handling"""
    # Handle negations first
    text = re.sub(r"(not\s+|no\s+|n't\s+|cannot\s+)", " not_", text, flags=re.IGNORECASE)
    
    # Standard cleaning
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters and spaces
    text = text.lower().strip()  # Convert to lowercase and trim
    text = re.sub(r'\s+', ' ', text)  # Collapse multiple spaces
    return text

def load_imdb_data(data_path):
    """Load raw IMDb data using sklearn's load_files"""
    reviews = load_files(data_path, 
                        categories=['pos', 'neg'],
                        encoding='utf-8',
                        shuffle=True,
                        random_state=42)
    return reviews.data, reviews.target

def prepare_data():
    """Prepare balanced 5k dataset from raw IMDb files"""
    try:
        # Path configuration
        raw_data_path = os.path.join('data', 'aclImdb', 'train')
        train_path = os.path.join('data', 'train_5k.csv')
        test_path = os.path.join('data', 'test_5k.csv')
        
        # Load and process raw data
        print("Loading raw IMDb data...")
        texts, labels = load_imdb_data(raw_data_path)
        
        # Create DataFrame
        df = pd.DataFrame({
            'review': texts,
            'sentiment': ['positive' if label == 1 else 'negative' for label in labels]
        })
        
        # Take balanced 5,000 samples (2,500 each)
        print("Sampling 5,000 reviews (2,500 pos/neg)...")
        pos_samples = df[df['sentiment'] == 'positive'].sample(2500, random_state=42)
        neg_samples = df[df['sentiment'] == 'negative'].sample(2500, random_state=42)
        df = pd.concat([pos_samples, neg_samples])
        
        # Clean and prepare data
        print("Cleaning text data...")
        df['review'] = df['review'].apply(clean_text)
        df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
        
        # Split and save (4,000 train + 1,000 test)
        print("Splitting into train/test sets...")
        train, test = train_test_split(df, test_size=0.2, random_state=42)
        
        # Ensure data directory exists
        os.makedirs('data', exist_ok=True)
        
        train.to_csv(train_path, index=False)
        test.to_csv(test_path, index=False)
        
        print("\nData preparation complete:")
        print(f"  Training samples: {len(train)} saved to {train_path}")
        print(f"  Test samples: {len(test)} saved to {test_path}")
        
    except Exception as e:
        print(f"\nError during data preparation: {str(e)}")
        print("Please verify:")
        print(f"1. Dataset is downloaded from http://ai.stanford.edu/~amaas/data/sentiment/")
        print(f"2. Extracted to ./data/aclImdb/ with train/pos and train/neg folders")

if __name__ == '__main__':
    prepare_data()
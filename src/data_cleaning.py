import pandas as pd
import re
from sklearn.model_selection import train_test_split

def clean_text(text):
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only letters
    text = text.lower()  # Convert to lowercase
    return text

def prepare_data():
    # Load full dataset
    df = pd.read_csv('.//data//imdb_dataset.csv')
    
    # Take balanced 5,000 samples (2,500 each)
    pos_samples = df[df['sentiment'] == 'positive'].sample(2500, random_state=42)
    neg_samples = df[df['sentiment'] == 'negative'].sample(2500, random_state=42)
    df = pd.concat([pos_samples, neg_samples])
    
    # Clean and prepare data
    df['review'] = df['review'].apply(clean_text)
    df['sentiment'] = df['sentiment'].map({'positive': 1, 'negative': 0})
    
    # Split and save (will give 4,000 train + 1,000 test)
    train, test = train_test_split(df, test_size=0.2, random_state=42)
    train.to_csv('.//data//train_5k.csv', index=False)
    test.to_csv('.//data//test_5k.csv', index=False)
    print(f"Saved {len(train)} training and {len(test)} test samples")
def clean_text(text):
    # Handle negations (important for sentiment)
    text = re.sub(r"(not\s+|no\s+|n't\s+)", " not_", text)
    
    # Standard cleaning
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep letters+spaces
    text = text.lower()
    return text    

if __name__ == '__main__':
    prepare_data()
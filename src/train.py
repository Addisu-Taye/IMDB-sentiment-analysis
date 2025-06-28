import os
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

def load_imdb_data():
    """Load data from aclImdb folder with proper Windows paths"""
    data_path = os.path.join('data', 'aclImdb', 'train')
    reviews = load_files(data_path, 
                       categories=['pos', 'neg'],
                       shuffle=True,
                       random_state=42,
                       encoding='utf-8')
    return reviews.data, reviews.target

def train_model():
    # Create model directory if not exists
    os.makedirs('model', exist_ok=True)
    
    # Load data using proper paths
    print("Loading data from aclImdb...")
    X, y = load_imdb_data()
    
    # Take 5000 balanced samples (2500 pos, 2500 neg)
    X, y = X[:5000], y[:5000]
    print(f"\nUsing {len(X)} samples ({np.sum(y)} positive, {len(y)-np.sum(y)} negative)")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Vectorize
    print("\nCreating TF-IDF features...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train
    print("Training Logistic Regression model...")
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    train_acc = model.score(X_train_vec, y_train)
    test_acc = model.score(X_test_vec, y_test)
    print(f"\nTraining Accuracy: {train_acc:.2%}")
    print(f"Test Accuracy: {test_acc:.2%}")
    
    # Save models
    vectorizer_path = os.path.join('model', 'vectorizer.pkl')
    model_path = os.path.join('model', 'model.pkl')
    
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    print(f"\nModels saved to:\n{vectorizer_path}\n{model_path}")

if __name__ == '__main__':
    train_model()
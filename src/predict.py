import pickle
import re  # <-- This was missing
from sklearn.feature_extraction.text import TfidfVectorizer

def load_models():
    """Load trained models"""
    with open('./model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('./model/model.pkl', 'rb') as f:
        model = pickle.load(f)
    return vectorizer, model

def predict_sentiment(text):
    """Predict sentiment with negation handling"""
    # Clean text (same as training)
    text = re.sub(r"(not\s+|no\s+|n't\s+|cannot\s+)", " not_", text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower().strip()
    
    # Load models and predict
    vectorizer, model = load_models()
    vec = vectorizer.transform([text])
    proba = model.predict_proba(vec)[0]
    pred = model.predict(vec)[0]
    
    return "positive" if pred == 1 else "negative", round(max(proba) * 100, 1)

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your review text\"")
        sys.exit(1)
    
    review_text = ' '.join(sys.argv[1:])
    sentiment, confidence = predict_sentiment(review_text)
    print(f"{sentiment} (confidence: {confidence}%)")
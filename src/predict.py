import pickle
import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def load_models():
    with open('model/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    with open('model/classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)
    return vectorizer, classifier

def predict_sentiment(text):
    # Add the same negation handling as cleaning
    text = re.sub(r"(not\s+|no\s+|n't\s+)", " not_", text)
    vectorizer, classifier = load_models()
    vec = vectorizer.transform([text])
    proba = classifier.predict_proba(vec)[0]
    prediction = classifier.predict(vec)[0]
    
    sentiment = 'positive' if prediction == 1 else 'negative'
    confidence = max(proba) * 100
    return sentiment, confidence

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python predict.py \"Your review text here\"")
        sys.exit(1)
    
    review_text = ' '.join(sys.argv[1:])
    sentiment, confidence = predict_sentiment(review_text)
    print(f"{sentiment} (confidence: {confidence:.1f}%)")
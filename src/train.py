from sklearn.datasets import load_files
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

# Create model directory if it doesn't exist
os.makedirs('model', exist_ok=True)

# Load dataset (using a small subset for demonstration)
# In a real scenario, you would load the actual IMDb dataset
reviews = load_files('./data/aclImdb/train', categories=['pos', 'neg'], 
                    shuffle=True, random_state=42, 
                    encoding='utf-8')

# Use only 5000 samples for this demo
X, y = reviews.data[:5000], reviews.target[:5000]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Vectorize text
vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train classifier
classifier = LogisticRegression(max_iter=1000)
#classifier.fit(X_train_vec, y_train)
classifier = LogisticRegression(
    max_iter=1000,
    class_weight='balanced',  # Helps with imbalanced data
    C=0.1  # Stronger regularization
)

# Evaluate
y_pred = classifier.predict(X_test_vec)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save models
with open('model/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)
    
with open('model/classifier.pkl', 'wb') as f:
    pickle.dump(classifier, f)
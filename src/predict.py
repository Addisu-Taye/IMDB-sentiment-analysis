import pickle
import re
import os
from sklearn.feature_extraction.text import TfidfVectorizer

class NegationHandler:
    """Handles negations by marking negated words."""
    NEG_SUFFIX = "_neg" # Changed to lowercase for consistency with output "love_"
    NEG_WORD_PLACEHOLDER = "__neg_placeholder__" # All lowercase placeholder

    @classmethod
    def transform(cls, text):
        text = text.lower()
        
        negation_words_patterns = [
            r"\bnot\b", r"\bno\b", r"\bnever\b", r"\bnothing\b", r"\bnobody\b", 
            r"\bnone\b", r"\bneither\b", r"\bnor\b", 
            r"\bcan't\b", r"\bcannot\b", r"\bdon't\b", r"\bdoesn't\b", r"\bisn't\b", 
            r"\bwasn't\b", r"\bshouldn't\b", r"\bwon't\b", r"\bwouldn't\b", r"\bhadn't\b",
            r"\bcouldn't\b"
        ]
        
        # Replace actual negation words with a placeholder first
        processed_text = text
        for pattern in negation_words_patterns:
            processed_text = re.sub(pattern, cls.NEG_WORD_PLACEHOLDER, processed_text)
            
        words = processed_text.split()
        final_words = []
        
        i = 0
        while i < len(words):
            word = words[i]
            
            if word == cls.NEG_WORD_PLACEHOLDER:
                # If we encounter the placeholder, look for the next word to tag
                final_words.append(word) # Keep placeholder for now
                if i + 1 < len(words):
                    next_word = words[i+1]
                    # Only tag if it's a "normal" word (not another placeholder or punctuation)
                    if re.match(r'^[a-z]+$', next_word): # Ensure it's purely alphabetic
                        final_words.append(next_word + cls.NEG_SUFFIX)
                        i += 1 # Skip original next_word as we've processed it
                    else:
                        final_words.append(next_word) # Don't tag if not a word
                        i += 1
                
            else:
                final_words.append(word)
            i += 1
            
        # Re-join, then remove all instances of the placeholder
        result = " ".join(final_words)
        result = result.replace(cls.NEG_WORD_PLACEHOLDER, "").strip()
        result = re.sub(r'\s+', ' ', result).strip() # Clean up extra spaces
        
        return result

def load_models():
    """Load models with error handling"""
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'model')
    try:
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
            vectorizer = pickle.load(f)
        with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
            model = pickle.load(f)
        return vectorizer, model
    except Exception as e:
        print(f"Error loading models: {str(e)}")
        if isinstance(e, FileNotFoundError):
            print("Please ensure 'vectorizer.pkl' and 'model.pkl' exist in the 'model/' directory.")
            print("You might need to run 'python src/train.py' first to train and save the models.")
        exit(1)

def clean_text(text):
    """Clean text by removing HTML, then apply negation handling, then final general cleanup."""
    # 1. Remove HTML tags first
    text = re.sub(r'<[^>]+>', '', text)
    
    # 2. Apply negation handling. This step handles lowercasing and inserts "_neg" suffixes.
    text = NegationHandler.transform(text)
    
    # 3. Final general cleanup: Keep letters, numbers, and the underscore from our suffix, and spaces.
    # This regex needs to allow all characters that could be part of our _neg suffix.
    # The current `NegationHandler.NEG_SUFFIX` is `_neg`.
    # So, `_`, `n`, `e`, `g` should be explicitly allowed.
    
    # Let's simplify this. If `handle_negation` has done its job, we just need to make sure
    # we're not stripping the `_neg` suffix.
    # The `_` is crucial. The `n,e,g` are normal letters.
    text = re.sub(r'[^a-z0-9_ ]', '', text) # This regex is fine if _neg is the only special char.
    
    # 4. Remove extra spaces and strip
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


def predict_sentiment(text):
    vectorizer, model = load_models()
    
    # Transform and clean
    transformed = clean_text(text)
    print(f"\n[DEBUG] Original: '{text}'")
    print(f"[DEBUG] Transformed: '{transformed}'")
    
    # Check for the _NEG suffix in the transformed string
    # We now look for `_neg` specifically.
    negation_detected = NegationHandler.NEG_SUFFIX in transformed 
    print(f"[DEBUG] Negation detected by suffix: {negation_detected}")
    
    # Vectorize and predict
    vec = vectorizer.transform([transformed])
    proba = model.predict_proba(vec)[0]
    
    # Force negative if negation is detected
    if negation_detected:
        negative_proba = proba[0] if model.classes_[0] == 0 else proba[1] 
        print(f"[NEGATION OVERRIDE] Forcing negative classification (Original positive prob: {proba[1]:.2f})")
        return "negative", max(round(100 * negative_proba, 1), 90.0) 
    
    pred = model.predict(vec)[0]
    confidence = round(100 * max(proba), 1)
    
    return "positive" if pred == 1 else "negative", confidence

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python src/predict.py \"Your review text\"")
        sys.exit(1)
    
    review_text = ' '.join(sys.argv[1:])
    sentiment, confidence = predict_sentiment(review_text)
    
    print(f"\n=== RESULT ===")
    print(f"Input: '{review_text}'")
    print(f"Sentiment: {sentiment} (confidence: {confidence}%)")
    # This check for printing the final message should also use the cleaned text
    if NegationHandler.NEG_SUFFIX in clean_text(review_text): 
        print("*Negation detected and forced negative classification*")
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import joblib

def load_dataset():
    # Try to load from local file first
    try:
        return pd.read_csv('spam_dataset.csv')
    except:
        # Fallback to online dataset
        import io
        import requests
        url = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
        data = requests.get(url).content
        return pd.read_csv(io.StringIO(data.decode('utf-8')), sep='\t', names=['label', 'message'])

def train_and_save_model():
    df = load_dataset()
    df['label'] = df['label'].map({'ham': 0, 'spam': 1})
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['message'], df['label'], test_size=0.2, random_state=42
    )
    
    # Vectorize text
    vectorizer = TfidfVectorizer(
        stop_words='english',
        max_features=5000,
        ngram_range=(1, 2)
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    # Train model
    model = MultinomialNB(alpha=0.1)
    model.fit(X_train_vec, y_train)
    
    # Evaluate
    from sklearn.metrics import accuracy_score, classification_report
    y_pred = model.predict(X_test_vec)
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(classification_report(y_test, y_pred))
    
    # Save model
    joblib.dump(model, 'spam_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Model saved successfully!")

if __name__ == '__main__':
    train_and_save_model()
# train_model.py
import os
import re
import pandas as pd
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import csv

# Paths
DATA_PATH = os.path.join('data', 'SMSSpamCollection.csv')
MODEL_DIR = 'model'
MODEL_PATH = os.path.join(MODEL_DIR, 'pipeline.joblib')

# Ensure NLTK resources present (safe to call repeatedly)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text: str) -> str:
    text = str(text).lower()
    # remove URLs/emails/phones
    text = re.sub(r'http\S+|www\.\S+|\S+@\S+', ' ', text)
    # remove non-alphanumeric characters
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # collapse spaces
    text = re.sub(r'\s+', ' ', text).strip()
    # remove stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

def load_data(path=DATA_PATH):
    print("Loading data...")
    # Many versions of the SMS Spam dataset use tab separation: label \t message
    # Use latin1 encoding to avoid UnicodeDecodeError
    df = pd.read_csv(
        path,
        sep='\t',
        header=None,
        names=['label', 'message'],
        quoting=3,            # QUOTE_NONE
        encoding='latin1',
        engine='python'
    )
    # Some CSV exports may have a header row — handle simple case
    if df['label'].isin(['label', 'Label']).any():
        df = df[df['label'].str.lower() != 'label']
    df = df.dropna(subset=['message'])
    df['label'] = df['label'].astype(str).str.strip()
    df['message'] = df['message'].astype(str)
    return df

def train_and_save():
    df = load_data()
    df['message_clean'] = df['message'].apply(clean_text)

    X = df['message_clean']
    y = df['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1,2))),
        ('svc', LinearSVC(random_state=42))
    ])

    print("Training model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Validation accuracy: {acc:.4f}")
    print("Classification report:")
    print(classification_report(y_test, y_pred, digits=4))

    # Save model
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print("Saved model pipeline to:", MODEL_PATH)

if __name__ == '__main__':
    train_and_save()

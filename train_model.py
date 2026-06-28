import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
import pickle
import re

print("Loading dataset...")
df = pd.read_csv('WELFake_Dataset.csv')
df = df.dropna(subset=['title', 'text', 'label'])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:500]

print("Cleaning text...")
df['combined'] = (df['title'] + ' ' + df['text']).apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['combined'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

print("Vectorizing...")
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    sublinear_tf=True,
    ngram_range=(1,2)
)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

print("Training...")
model = LogisticRegression(max_iter=1000, C=1.5, class_weight='balanced')
model.fit(X_train_vec, y_train)

pred = model.predict(X_test_vec)
print("Accuracy:", round(accuracy_score(y_test, pred) * 100, 2), "%")
print(classification_report(y_test, pred, target_names=['REAL','FAKE']))

pickle.dump(model, open('model.pkl', 'wb'))
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
print("Done!")

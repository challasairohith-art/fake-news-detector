import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import pickle
import re

df = pd.read_csv('WELFake_Dataset.csv')
df = df.dropna(subset=['title', 'text', 'label'])

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text[:500]

df['combined'] = (df['title'] + ' ' + df['text']).apply(clean_text)

X_train, X_test, y_train, y_test = train_test_split(
    df['combined'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)

model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

X_test_vec = vectorizer.transform(X_test)
y_pred = model.predict(X_test_vec)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['REAL', 'FAKE'])
disp.plot(cmap='Blues')
plt.title('Fake News Detector - Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Confusion matrix saved!")

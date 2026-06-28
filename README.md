# Fake News Detector

An AI-powered web app that detects fake news using Machine Learning.

## Features
- 94.94% accuracy on 72,000+ news articles
- Built from scratch using TF-IDF + Logistic Regression
- REST API built with Flask
- Clean web interface

## Tech Stack
- Python, Scikit-learn, Flask
- TF-IDF Vectorizer (5000 features)
- WELFake Dataset (72,134 articles)

## How to Run
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Download WELFake dataset from Kaggle and place in root folder
4. Train the model: `python3 train_model.py`
5. Run the app: `python3 app.py`
6. Open `http://127.0.0.1:5000`

## Model Performance
- Accuracy: 94.94%
- Precision: 95% (REAL), 94% (FAKE)
- Recall: 93% (REAL), 97% (FAKE)

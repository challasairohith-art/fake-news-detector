from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)

print("Loading model...")
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
print("Model loaded!")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    statement = data.get('statement', '')

    if not statement:
        return jsonify({'error': 'No statement provided'}), 400

    vectorized = vectorizer.transform([statement])
    prediction = model.predict(vectorized)[0]
    probability = model.predict_proba(vectorized)[0]
    confidence = round(max(probability) * 100, 2)

    label = 'FAKE' if int(prediction) == 1 else 'REAL'

    return jsonify({
        'statement': statement,
        'prediction': label,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(debug=False)

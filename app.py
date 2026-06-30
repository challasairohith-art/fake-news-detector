from flask import Flask, request, jsonify, render_template, session, redirect, url_for
from flask_mail import Mail, Message
import pickle
import random
import os

app = Flask(__name__)
app.secret_key = 'fakenews2026secretkey'

app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'fakenewsdetector.otp@gmail.com'
app.config['MAIL_PASSWORD'] = 'tdhrltmfnqgjiibw'

mail = Mail(app)

print("Loading model...")
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
print("Model loaded!")

otp_store = {}

@app.route('/login')
def login():
    if 'email' in session:
        return redirect(url_for('home'))
    return render_template('login.html')

@app.route('/send-otp', methods=['POST'])
def send_otp():
    data = request.get_json()
    email = data.get('email', '')
    if not email:
        return jsonify({'success': False, 'message': 'Email is required'})
    otp = str(random.randint(100000, 999999))
    otp_store[email] = otp
    try:
        msg = Message(
            subject='Your OTP - Fake News Detector',
            sender='fakenewsdetector.otp@gmail.com',
            recipients=[email]
        )
        msg.body = f'''Hello!

Your OTP for Fake News Detector is: {otp}

This OTP is valid for 5 minutes. Do not share it with anyone.

Regards,
Fake News Detector Team'''
        mail.send(msg)
        return jsonify({'success': True, 'message': 'OTP sent successfully'})
    except Exception as e:
        return jsonify({'success': False, 'message': 'Failed to send OTP. Try again.'})

@app.route('/verify-otp', methods=['POST'])
def verify_otp():
    data = request.get_json()
    email = data.get('email', '')
    otp = data.get('otp', '')
    if email in otp_store and otp_store[email] == otp:
        session['email'] = email
        del otp_store[email]
        return jsonify({'success': True})
    return jsonify({'success': False, 'message': 'Invalid OTP. Please try again.'})

@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('login'))

@app.route('/')
def home():
    if 'email' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', email=session['email'])

@app.route('/predict', methods=['POST'])
def predict():
    if 'email' not in session:
        return jsonify({'error': 'Unauthorized'}), 401
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

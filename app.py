from flask import Flask, render_template, request, redirect, url_for, session, flash
from pymongo import MongoClient
import pandas as pd
import joblib
import os
from datetime import datetime
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)

# MongoDB Configuration
client = MongoClient(
    'Paste Your mongodb url',
    tlsAllowInvalidCertificates=True  # Disables SSL certificate verification
)
db = client['spamshield']
detections = db['detections']
contacts = db['contacts']

# Load ML model
model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/manual', methods=['GET', 'POST'])
def manual():
    if request.method == 'POST':
        message = request.form.get('message')
        if not message:
            flash('Please enter a message', 'error')
            return redirect(url_for('manual'))
        
        # Predict
        message_vec = vectorizer.transform([message])
        prediction = model.predict(message_vec)[0]
        proba = model.predict_proba(message_vec)[0][prediction]
        
        # Store result
        detections.insert_one({
            'type': 'manual',
            'message': message,
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': float(proba),
            'timestamp': datetime.now()
        })
        
        session['last_result'] = {
            'message': message,
            'prediction': 'spam' if prediction == 1 else 'ham',
            'confidence': float(proba)
        }
        return redirect(url_for('results'))
    
    return render_template('manual.html')

@app.route('/batch', methods=['GET', 'POST'])
def batch():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded', 'error')
            return redirect(url_for('batch'))
        
        file = request.files['file']
        if file.filename == '':
            flash('No file selected', 'error')
            return redirect(url_for('batch'))
        
        try:
            # Read file
            if file.filename.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            
            if 'message' not in df.columns:
                flash('File must contain "message" column', 'error')
                return redirect(url_for('batch'))
            
            # Process messages
            messages = df['message'].fillna('').astype(str)
            message_vecs = vectorizer.transform(messages)
            predictions = model.predict(message_vecs)
            probabilities = model.predict_proba(message_vecs).max(axis=1)
            
            # Store results
            batch_results = []
            for msg, pred, prob in zip(messages, predictions, probabilities):
                batch_results.append({
                    'message': msg,
                    'prediction': 'spam' if pred == 1 else 'ham',
                    'confidence': float(prob)
                })
                detections.insert_one({
                    'type': 'batch',
                    'message': msg,
                    'prediction': 'spam' if pred == 1 else 'ham',
                    'confidence': float(prob),
                    'timestamp': datetime.now()
                })
            
            session['batch_results'] = batch_results
            return redirect(url_for('results'))
        
        except Exception as e:
            flash(f'Error processing file: {str(e)}', 'error')
            return redirect(url_for('batch'))
    
    return render_template('batch.html')

@app.route('/results')
def results():
    manual_result = session.pop('last_result', None)
    batch_results = session.pop('batch_results', None)
    return render_template('results.html',
                         manual_result=manual_result,
                         batch_results=batch_results)

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        message = request.form.get('message')
        
        # Store in MongoDB
        contacts.insert_one({
            'name': name,
            'email': email,
            'message': message,
            'timestamp': datetime.now(),
            'status': 'unread'
        })
        
        flash('Thank you for your message! We will get back to you soon.', 'success')
        return redirect(url_for('contact'))
    
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
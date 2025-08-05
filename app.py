from flask import Flask, request, jsonify, render_template
import os
import pickle
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import nltk

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Load model and vectorizer
model = pickle.load(open('models/lr_bow.pkl', 'rb'))
vectorizer = pickle.load(open('models/cv_vectorizer.pkl', 'rb'))

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def clean_text(text):
    text = re.sub(r"[^a-zA-Z]", " ", str(text).lower())
    words = text.split()
    words = [stemmer.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# ✅ Serve HTML file from root
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Single text prediction
    if request.is_json:
        data = request.get_json()
        if 'text' in data:
            input_text = data['text']
            cleaned = clean_text(input_text)
            vec = vectorizer.transform([cleaned])
            result = model.predict(vec)[0]
            prediction = 'Positive 😊' if result == 1 else 'Negative 😠'
            return jsonify({'prediction': prediction})

    # CSV bulk prediction
    elif 'file' in request.files:
        file = request.files['file']
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            if df.empty or df.shape[1] < 1:
                return "Invalid or empty file", 400

            column = df.columns[0]
            df = df.dropna()
            df['cleaned'] = df[column].apply(clean_text)
            vectors = vectorizer.transform(df['cleaned'])
            df['prediction'] = model.predict(vectors)
            df['prediction'] = df['prediction'].apply(lambda x: 'Positive 😊' if x == 1 else 'Negative 😠')

            positive = (df['prediction'] == 'Positive 😊').sum()
            negative = (df['prediction'] == 'Negative 😠').sum()

            return jsonify({
                'positive': int(positive),
                'negative': int(negative),
                'predictions': df['prediction'].tolist()
            })

        else:
            return "Only .csv files are supported", 400

    return "Invalid request", 400

if __name__ == '__main__':
    app.run(debug=True)

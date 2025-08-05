🎬 Sentiment Analysis on IMDb Movie Reviews
This project is a machine learning-based sentiment analysis system built using scikit-learn and Flask, which predicts whether a given movie review is positive or negative. The model is trained on the IMDb dataset containing 50,000 labeled reviews (balanced with 25k positive and 25k negative).

🚀 Demo
📌 Enter a movie review and get a real-time sentiment prediction (Positive/Negative) from the web interface.

📁 Project Structure
.
├── app.py                  # Flask app for prediction
├── models/
│   ├── lr_tfidf.pkl        # Trained Logistic Regression model
│   └── tv.pkl              # TF-IDF vectorizer
├── templates/
│   └── index.html          # Frontend HTML template
├── static/
│   └── style.css           # (optional) styling for frontend
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation

📦 Features
Text preprocessing (cleaning, stemming, stopword removal)

Sentiment prediction using:

CountVectorizer

TfidfVectorizer

Models used:

Logistic Regression (main)

Multinomial Naive Bayes

SVM (optional)

Flask web app for easy interaction

🛠️ Installation
Clone the repository

bash
Copy
Edit
git clone https://github.com/NandaKishore1316/<REPO_NAME>.git
cd <REPO_NAME>
Create a virtual environment (optional but recommended)

bash
Copy
Edit
python -m venv venv
venv\Scripts\activate   # On Windows
Install dependencies

bash
Copy
Edit
pip install -r requirements.txt
▶️ Run the App
bash
Copy
Edit
python app.py
Then open your browser and go to:
http://127.0.0.1:5000

🧠 Model Training
The model was trained using:

TF-IDF features

Logistic Regression

Accuracy: ~89% on test data

You can re-train using the IMDb dataset:

Link: IMDb Dataset

📊 Visualization (Optional)
WordCloud for positive/negative reviews

Sentiment distribution using Matplotlib/Seaborn

📚 Dependencies
Core libraries used:

text
Copy
Edit
Flask
scikit-learn
pandas
nltk
textblob
matplotlib
wordcloud
bs4


👨‍💻 Author
Nanda Kishore


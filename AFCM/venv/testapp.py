from flask import Flask, render_template, request
import os
import joblib
import re
import pdfplumber
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datetime import datetime
import subprocess
from glob import glob
from flask_cors import CORS
from flask import jsonify

# Download NLTK resources (run once)
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

app = Flask(__name__)

# Load the latest machine learning model (pipeline)
existing_models = glob(os.path.join(os.path.dirname(__file__), 'model', 'AFCM_pipeline*.joblib'))
latest_counts = [int(re.search(r'\d+', model).group()) for model in existing_models if re.search(r'\d+', model) is not None]
latest_count = max(latest_counts, default=0)
latest_model_path = os.path.join(os.path.dirname(__file__), 'model', f'AFCM_pipeline{latest_count}.joblib')
pipeline = joblib.load(latest_model_path, mmap_mode='r')

# Set the secret key to secure your sessions
app.config['SECRET_KEY'] = 'your_secret_key'

# Set the template and static file folders
app.template_folder = 'templates'
app.static_folder = 'static'

# Enable CORS for all routes
CORS(app)

def remove_stops(text):
    # Use regular expression to match and remove the dynamic pattern
    text = re.sub(r'M\d+_GADD\d+_\d+_SE_C01\.QXD \d+/\d+/\d+ \d+:\d+ [APMapm]{2} Page \d+', '', text)

    # Remove line breaks
    text = text.replace("\n", "")

    return text.strip()  # Remove leading and trailing whitespaces

def preprocess_text(text):
    # Tokenization and lemmatization
    stop_words_custom = set(["would", "could", "should", "might", "must", "shall"])  # Add more as needed
    stop_words = stop_words_custom.union(set(stopwords.words('english')))
    
    lemmatizer = WordNetLemmatizer()
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token.lower()) for token in tokens if token.isalnum() and token.lower() not in stop_words]

    return ' '.join(tokens)

def extract_keywords(text):
    # Use TF-IDF for feature extraction with custom stop words
    stop_words_custom = set(["would", "could", "should", "might", "must", "shall"])  # Add more as needed
    stop_words = stop_words_custom.union(set(stopwords.words('english')))
    
    vectorizer = TfidfVectorizer(max_features=150, stop_words=list(stop_words))
    X = vectorizer.fit_transform([text])

    # Select the top 5 keywords using chi-squared test
    selector = SelectKBest(chi2, k=5)
    selector.fit(X, [0])  # Assuming a single class for simplicity

    # Get the feature names (keywords)
    feature_names = vectorizer.get_feature_names_out()
    
    # Get the indices of the top 5 features
    top_indices = selector.get_support(indices=True)

    # Extract the top 5 keywords
    top_keywords = [feature_names[i] for i in top_indices]

    return top_keywords

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Check if the request contains a file with the name 'pdf_file'
        if 'pdf_file' in request.files:
            pdf_file = request.files['pdf_file']
            
            # Print the file name for debugging
            print(f"Received file: {pdf_file.filename}")
            
            # Check if the file is a PDF
            if pdf_file and pdf_file.filename.endswith('.pdf'):
                # Read the content of the PDF file as text
                pdf_content = ""
                for page in pdfplumber.open(pdf_file).pages:
                    pdf_content += page.extract_text()

                # Preprocess the text
                cleaned_text = preprocess_text(remove_stops(pdf_content))

                # Extract keywords
                keywords = extract_keywords(cleaned_text)

                # Make predictions using the loaded pipeline
                predicted_discipline = pipeline.predict([cleaned_text])

                # Return the prediction and keywords
                return jsonify({'discipline': predicted_discipline[0], 'keywords': keywords})

        # If no valid file is provided or an error occurs
        return jsonify({'error': 'Invalid file format'})

def train_svm_model():
    try:
        # Check if the current date is the 20th of December
        current_date = datetime.now()
        if current_date.month == 12 and current_date.day == 20:
            # Specify the path to svmtraining.py
            svm_script_path = 'C:/Users/LENOVO/Desktop/afcmflask/AFCM/venv/svmtraining.py'

            # Run the svmtraining.py script using subprocess
            subprocess.run(['python', svm_script_path])

            print("SVM model re-training completed successfully.")
        else:
            print("SVM model training skipped. Today is not the 20th of December.")
    except Exception as e:
        print(f"Error during SVM model training: {e}")
        
# Train the SVM model only on the 20th of December
train_svm_model()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
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

# Download NLTK resources (run once)
# import nltk
# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')

app = Flask(__name__)

# Load the machine learning model (pipeline)
pipeline_path = os.path.join(os.path.dirname(__file__), 'model', 'AFCM_pipeline.joblib')
pipeline = joblib.load(pipeline_path, mmap_mode='r')

# Set the secret key to secure your sessions
app.config['SECRET_KEY'] = 'your_secret_key'

# Set the template and static file folders
app.template_folder = 'templates'
app.static_folder = 'static'

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
                return render_template('index.html', prediction=predicted_discipline[0], keywords=keywords)

        # If no valid file is provided or an error occurs
        return render_template('index.html', prediction="Error: Invalid file format")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from docx import Document
import fitz  # PyMuPDF for PDF handling

app = Flask(__name__)

# Directory paths
STORED_FILES_DIR = 'stored_files'
UPLOADED_FILES_DIR = 'uploaded_files'

# Ensure the uploaded_files directory exists
os.makedirs(UPLOADED_FILES_DIR, exist_ok=True)
# Ensure the directory exists
os.makedirs(STORED_FILES_DIR, exist_ok=True)
# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx'}

# Route: Homepage
@app.route('/')
def index():
    return render_template('index.html')

# Route: Handle File Upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part in the request", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    # Ensure file extension is allowed
    if not allowed_file(file.filename):
        return "File type not allowed", 400

    # Save uploaded file
    uploaded_file_path = os.path.join(UPLOADED_FILES_DIR, file.filename)
    file.save(uploaded_file_path)

    # Compare with stored files
    results = compare_files(uploaded_file_path)

    # Render results page
    return render_template('results.html', results=results)

# Function: Check if file extension is allowed
# Function to check allowed file types
def allowed_file(filename):
    allowed_extensions = {'txt', 'pdf', 'docx'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


# Function: Extract text from a .docx file
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text

# Function: Extract text from a .pdf file
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text("text")  # Extract text from each page
    return text

def extract_text(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.txt':
        with open(file_path, 'r') as file:
            return file.read()
    else:
        # Log unsupported file types
        print(f"Unsupported file type detected: {file_extension}")
        raise ValueError(f"Unsupported file type: {file_extension}!")


# Function: Compare Files Using Cosine Similarity
def compare_files(uploaded_file_path):
    uploaded_text = extract_text(uploaded_file_path)

    stored_texts = []
    file_names = []

    # Load stored files and log any issues
    for filename in os.listdir(STORED_FILES_DIR):
        stored_file_path = os.path.join(STORED_FILES_DIR, filename)
        
        # Debugging statement to print the file extension and name
        print(f"Processing stored file: {filename}")
        
        try:
            stored_text = extract_text(stored_file_path)
            stored_texts.append(stored_text)
            file_names.append(filename)
        except ValueError as e:
            print(f"Error processing file {filename}: {str(e)}")

    # Combine uploaded text with stored texts
    texts = [uploaded_text] + stored_texts

    # Convert texts to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Prepare results
    results = [{'file': file_names[i], 'similarity': round(similarity_scores[i] * 100, 2)} for i in range(len(file_names))]
    return sorted(results, key=lambda x: x['similarity'], reverse=True)


if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)


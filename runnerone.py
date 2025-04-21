from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import fitz  # PyMuPDF for PDF handling
import pytesseract
from PIL import Image
from docx import Document
import aiohttp
import asyncio
from bs4 import BeautifulSoup

app = Flask(__name__)

# Directory paths
STORED_FILES_DIR = 'stored_files'
UPLOADED_FILES_DIR = 'uploaded_files'

# Ensure directories exist
os.makedirs(UPLOADED_FILES_DIR, exist_ok=True)
os.makedirs(STORED_FILES_DIR, exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg'}

# URLs to compare against
COMPARISON_URLS = [
    "https://www.bbc.com/news",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://medium.com/machine-learning",
    "https://arxiv.org/",
    "https://www.geeksforgeeks.org/python-programming-language/",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
    "https://www.reuters.com/",
    "https://www.sciencedirect.com/",
    "https://pubmed.ncbi.nlm.nih.gov/",
    "https://ieeexplore.ieee.org/",
    "https://www.academia.edu/",
    "https://stackoverflow.com/questions/tagged/python",
    "https://css-tricks.com/",
    "https://hackernoon.com/",
    "https://docs.oracle.com/en/java/",
    "https://flask.palletsprojects.com/en/2.2.x/",
    "https://www.tensorflow.org/",
    "https://towardsdatascience.com/",
    "https://machinelearningmastery.com/",
    "https://www.kaggle.com/",
    "https://realpython.com/"
]

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

    if not allowed_file(file.filename):
        return "File type not allowed", 400

    # Save uploaded file
    uploaded_file_path = os.path.join(UPLOADED_FILES_DIR, file.filename)
    file.save(uploaded_file_path)

    # Compare with stored files and web content
    results = asyncio.run(compare_files(uploaded_file_path))

    return render_template('results.html', results=results)

# Function: Check if file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Function: Extract text from DOCX
def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])

# Function: Extract text from PDF (with OCR fallback)
def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""

    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():
            text += page_text
        else:
            # If no text found, use OCR (for scanned PDFs)
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img)

    return text.strip()

# Function: Extract text from images (JPG, PNG)
def extract_text_from_image(file_path):
    return pytesseract.image_to_string(Image.open(file_path)).strip()

# Function: Extract text from any file type
def extract_text(file_path):
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.docx':
        return extract_text_from_docx(file_path)
    elif file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            return file.read()
    elif file_extension in ['.png', '.jpg', '.jpeg']:
        return extract_text_from_image(file_path)
    else:
        print(f"Unsupported file type detected: {file_extension}")
        raise ValueError(f"Unsupported file type: {file_extension}!")

# Asynchronous Function: Fetch and extract text from a URL
async def fetch_text_from_url(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                soup = BeautifulSoup(await response.text(), 'html.parser')
                for script_or_style in soup(['script', 'style']):
                    script_or_style.decompose()
                text = soup.get_text(separator=' ')
                return url, ' '.join(text.split())
            else:
                print(f"Failed to fetch {url}: Status code {response.status}")
                return url, ""
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return url, ""

# Asynchronous Function: Fetch all URLs concurrently
async def fetch_all_urls():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_text_from_url(session, url) for url in COMPARISON_URLS]
        return await asyncio.gather(*tasks)

# Function: Compare Files Using Cosine Similarity
async def compare_files(uploaded_file_path):
    uploaded_text = extract_text(uploaded_file_path)
    if not uploaded_text.strip():
        return [{"file": "No valid text found in uploaded file", "similarity": 0.0}]

    stored_texts = []
    file_names = []

    # Compare with stored files
    for filename in os.listdir(STORED_FILES_DIR):
        stored_file_path = os.path.join(STORED_FILES_DIR, filename)
        print(f"Processing stored file: {filename}")  # Debugging log

        try:
            stored_text = extract_text(stored_file_path)
            if stored_text.strip():
                stored_texts.append(stored_text)
                file_names.append(filename)
        except ValueError as e:
            print(f"Error processing file {filename}: {str(e)}")

    # Fetch and compare with web content concurrently
    web_results = await fetch_all_urls()
    for url, web_text in web_results:
        if web_text:
            stored_texts.append(web_text)
            file_names.append(url)

    if not stored_texts:
        return [{"file": "No stored files or web content with valid text found", "similarity": 0.0}]

    # Combine uploaded text with stored texts and web content
    texts = [uploaded_text] + stored_texts

    # Convert texts to TF-IDF vectors
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)

    # Calculate cosine similarity
    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    # Prepare results
    results = [{'file': file_names[i], 'similarity': round(similarity_scores[i] * 100, 2)} for i in range(len(file_names))]
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

# Run Flask App
if __name__ == '__main__':
    print("Starting Flask app...")
    app.run(debug=True)

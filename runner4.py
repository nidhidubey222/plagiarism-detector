from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from docx import Document
import aiohttp
import asyncio
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download NLTK data
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

app = Flask(__name__)

STORED_FILES_DIR = 'stored_files'
UPLOADED_FILES_DIR = 'uploaded_files'

os.makedirs(UPLOADED_FILES_DIR, exist_ok=True)
os.makedirs(STORED_FILES_DIR, exist_ok=True)

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'docx', 'png', 'jpg', 'jpeg', 'csv', 'xlsx'}
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "File type not allowed", 400

    uploaded_path = os.path.join(UPLOADED_FILES_DIR, file.filename)
    file.save(uploaded_path)

    results = asyncio.run(compare_files(uploaded_path))
    return render_template('results.html', results=results)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.docx':
        return "\n".join([para.text for para in Document(file_path).paragraphs])
    elif ext == '.pdf':
        return extract_text_from_pdf(file_path)
    elif ext == '.txt':
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif ext in ['.png', '.jpg', '.jpeg']:
        return pytesseract.image_to_string(Image.open(file_path)).strip()
    elif ext == '.csv':
        try:
            df = pd.read_csv(file_path)
            return '\n'.join(df.astype(str).apply(' '.join, axis=1))
        except Exception as e:
            return f"Error reading CSV: {e}"
    elif ext == '.xlsx':
        try:
            df = pd.read_excel(file_path, sheet_name=None)
            content = ''
            for sheet, data in df.items():
                content += '\n'.join(data.astype(str).apply(' '.join, axis=1))
            return content
        except Exception as e:
            return f"Error reading Excel: {e}"
    else:
        raise ValueError("Unsupported file type.")

def extract_text_from_pdf(file_path):
    text = ""
    doc = fitz.open(file_path)
    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():
            text += page_text
        else:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img)
    return text.strip()

def preprocess_text(text):
    words = word_tokenize(text.lower())
    cleaned = [lemmatizer.lemmatize(word) for word in words if word.isalpha() and word not in stop_words]
    return ' '.join(cleaned)

async def fetch_text_from_url(session, url):
    try:
        async with session.get(url) as response:
            if response.status == 200:
                soup = BeautifulSoup(await response.text(), 'html.parser')
                for tag in soup(['script', 'style']):
                    tag.decompose()
                return url, ' '.join(soup.get_text(separator=' ').split())
            else:
                return url, ""
    except Exception as e:
        return url, ""

async def fetch_all_urls():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_text_from_url(session, url) for url in COMPARISON_URLS]
        return await asyncio.gather(*tasks)

def duckduckgo_web_search(query, max_results=5):
    results = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            link = r.get('href') or r.get('url')
            if link:
                results.append(link)
    return results

async def fetch_duckduckgo_contents(query):
    urls = duckduckgo_web_search(query, max_results=5)
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_text_from_url(session, url) for url in urls]
        return await asyncio.gather(*tasks)

async def compare_files(uploaded_file_path):
    uploaded_text = preprocess_text(extract_text(uploaded_file_path))
    if not uploaded_text.strip():
        return [{"file": "No valid text found.", "similarity": 0.0}]

    texts = [uploaded_text]
    sources = []

    for filename in os.listdir(STORED_FILES_DIR):
        try:
            path = os.path.join(STORED_FILES_DIR, filename)
            text = preprocess_text(extract_text(path))
            if text.strip():
                texts.append(text)
                sources.append(filename)
        except Exception as e:
            print(f"Skipping {filename}: {e}")

    web_results = await fetch_all_urls()
    for url, text in web_results:
        processed = preprocess_text(text)
        if processed:
            texts.append(processed)
            sources.append(url)

    duck_results = await fetch_duckduckgo_contents(uploaded_text[:200])
    for url, text in duck_results:
        processed = preprocess_text(text)
        if processed:
            texts.append(processed)
            sources.append(url)

    if len(texts) == 1:
        return [{"file": "No valid comparisons found.", "similarity": 0.0}]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(texts)
    scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    result = [{'file': sources[i], 'similarity': round(scores[i] * 100, 2)} for i in range(len(scores))]
    return sorted(result, key=lambda x: x['similarity'], reverse=True)

if __name__ == '__main__':
    print("Running Flask app...")
    app.run(debug=True)

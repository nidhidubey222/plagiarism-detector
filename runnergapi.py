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
import requests

# Initialize Flask app
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
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript"
]

 #Google API Configuration
GOOGLE_API_KEY = "AIzaSyA2svMA667YBHtY6H5ByjAQw4dOxGKfK3o"  # Replace with your actual key
SEARCH_ENGINE_ID = "d1f1416e8fbf24c9f"  # Replace with your actual ID


@app.route('/')
def index():
    return render_template('index.html')


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

    # Compare with stored files, URLs, and Google API
    results = asyncio.run(compare_files(uploaded_file_path))

    return render_template('result.html', results=results)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def extract_text_from_docx(file_path):
    doc = Document(file_path)
    return "\n".join([para.text for para in doc.paragraphs])


def extract_text_from_pdf(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        page_text = page.get_text("text")
        if page_text.strip():
            text += page_text
        else:
            pix = page.get_pixmap()
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text += pytesseract.image_to_string(img)
    return text.strip()


def extract_text_from_image(file_path):
    return pytesseract.image_to_string(Image.open(file_path)).strip()


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
        raise ValueError(f"Unsupported file type: {file_extension}!")


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
                return url, ""
    except Exception as e:
        return url, ""


async def fetch_all_urls():
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_text_from_url(session, url) for url in COMPARISON_URLS]
        return await asyncio.gather(*tasks)


def google_search(query, num_results=5):
    search_url = "https://www.googleapis.com/customsearch/v1"
    
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": SEARCH_ENGINE_ID,
        "num": num_results
    }
    
    try:
        response = requests.get(search_url, params=params)
        response_json = response.json()

        # Debug: Print Google API response
        print("Google API Response:", response_json)

        if "items" not in response_json:
            print("No search results found or API error!")
            return []

        search_results = []
        for item in response_json["items"]:
            search_results.append({
                "title": item["title"],
                "link": item["link"],
                "snippet": item.get("snippet", "")
            })

        return search_results

    except requests.exceptions.RequestException as e:
        print("Error fetching Google Search results:", str(e))
        return []

async def compare_files(uploaded_file_path):
    uploaded_text = extract_text(uploaded_file_path)
    if not uploaded_text.strip():
        return [{"file": "No valid text found in uploaded file", "similarity": 0.0}]

    stored_texts = []
    file_names = []

    for filename in os.listdir(STORED_FILES_DIR):
        stored_file_path = os.path.join(STORED_FILES_DIR, filename)
        try:
            stored_text = extract_text(stored_file_path)
            if stored_text.strip():
                stored_texts.append(stored_text)
                file_names.append(filename)
        except ValueError:
            pass

    web_results = await fetch_all_urls()
    for url, web_text in web_results:
        if web_text:
            stored_texts.append(web_text)
            file_names.append(url)

    google_results = google_search(uploaded_text[:300])
    for result in google_results:
        stored_texts.append(result["snippet"])
        file_names.append(result["link"])

    if not stored_texts:
        return [{"file": "No stored files, web content, or Google results found", "similarity": 0.0}]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([uploaded_text] + stored_texts)

    similarity_scores = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    results = [{'file': file_names[i], 'similarity': round(similarity_scores[i] * 100, 2)} for i in range(len(file_names))]
    return sorted(results, key=lambda x: x['similarity'], reverse=True)


if __name__ == '__main__':
    app.run(debug=True)

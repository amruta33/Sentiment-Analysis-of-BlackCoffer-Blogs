from flask import Flask, render_template, request, jsonify
import requests
from bs4 import BeautifulSoup
from pathlib import Path
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import os
import re
import time

# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')

REQUIRENT_FOLDER_PATH = Path(__file__).parent.parent / "Blackcofee assignment"
STOP_WORDS_FOLDER_PATH = REQUIRENT_FOLDER_PATH / 'StopWords'
EXTRACTED_ARTICLES_FOLDER = REQUIRENT_FOLDER_PATH / 'extracted_articles'
MASTER_DIC_FOLDER_PATH = REQUIRENT_FOLDER_PATH / 'MasterDictionary'

app = Flask(__name__)

# Function to remove common sections from the text
def remove_common_section(text):
    patterns = [
        r'Project Snapshots.*?(?=Summarize|Contact Details)',
        r'Contact Details.*',
        r'Project website url\s*https?://\S+',
        r'Summarize.*?This project was done by the Blackcoffer Team, a Global IT Consulting firm\.',
        r'For project discussions and daily updates, would you like to use Slack, Skype, Telegram, or Whatsapp\? Please recommend, what would work best for you\.'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)
    return text.strip()

# Function to extract title and content from a URL
def extract_article_title_and_content(url, url_id, EXTRACTED_ARTICLES_FOLDER):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
    }
    for attempt in range(5):
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            soup = BeautifulSoup(response.content, 'html.parser')

            title_tag = soup.find('h1', {'class': 'entry-title'})
            title = title_tag.get_text(strip=True) if title_tag else "No Title Found"

            div_tag = soup.find('div', {'class': 'td-post-content tagdiv-type'})
            content = div_tag.get_text(separator="\n", strip=True) if div_tag else "Content not found."

            content = remove_common_section(content)

            combined_content = f"{title}\n{content}" if content else "Content not found."

            file_path = EXTRACTED_ARTICLES_FOLDER / f"{url_id}.txt"
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(combined_content)

            return combined_content

        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} - Error fetching URL {url}: {e}")
            time.sleep(5)
        except Exception as e:
            print(f"Attempt {attempt + 1} - An unexpected error occurred: {e}")
            time.sleep(5)

    return None

# Function to clean text using stopwords
def clean_with_stopwords(text, STOP_WORDS_FOLDER_PATH):
    stop_words = set()
    try:
        for file_path in os.listdir(STOP_WORDS_FOLDER_PATH):
            if file_path.endswith(".txt"):
                filepath = os.path.join(STOP_WORDS_FOLDER_PATH, file_path)
                with open(filepath, 'r', encoding='latin-1') as file:
                    for line in file:
                        if not re.match(r'http[s]?://', line.strip().lower()):
                            words = [word.strip().lower() for word in line.split('|') if word.strip()]
                            stop_words.update(words)
    except Exception as e:
        print(f"Error reading stopwords: {e}")
        return text

    text_without_urls = re.sub(r'http[s]?://\S+', '', text).replace("|", "")
    words = re.findall(r'\b\w+\b', text_without_urls.lower())
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words).strip()

# Function to create positive and negative word dictionaries
def create_positive_negative_dicts(MASTER_DIC_FOLDER_PATH, STOP_WORDS_FOLDER_PATH):
    def load_words_from_file(file_path):
        words = set()
        try:
            with open(file_path, 'r', encoding='latin-1') as file:
                for line in file:
                    word = line.strip().lower()
                    if word:
                        words.add(word)
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
        return words

    stop_words = set()
    for file_name in os.listdir(STOP_WORDS_FOLDER_PATH):
        if file_name.endswith(".txt"):
            file_path = os.path.join(STOP_WORDS_FOLDER_PATH, file_name)
            stop_words.update(load_words_from_file(file_path))

    positive_words = load_words_from_file(os.path.join(MASTER_DIC_FOLDER_PATH, "positive-words.txt")) - stop_words
    negative_words = load_words_from_file(os.path.join(MASTER_DIC_FOLDER_PATH, "negative-words.txt")) - stop_words
    

    return {
        "positive": positive_words,
        "negative": negative_words
    }

# Load positive and negative words at startup
word_dicts = create_positive_negative_dicts(MASTER_DIC_FOLDER_PATH, STOP_WORDS_FOLDER_PATH)
positive_words = word_dicts['positive']
negative_words = word_dicts['negative']
print(len(positive_words))
print(len(negative_words))
def calculate_scores(text):
    words = re.findall(r'\b\w+\b', text)
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)
    return positive_score, negative_score, polarity_score, subjectivity_score

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_blog():
    url = request.form.get('url')
    if not url:
        return jsonify({"error": "No URL provided"}), 400

    import hashlib
    url_id = hashlib.md5(url.encode()).hexdigest()

    EXTRACTED_ARTICLES_FOLDER.mkdir(parents=True, exist_ok=True)

    try:
        combined_content = extract_article_title_and_content(url, url_id, EXTRACTED_ARTICLES_FOLDER)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    if not combined_content:
        return jsonify({"error": "Unable to fetch content from the URL"}), 400

    cleaned_text = clean_with_stopwords(combined_content, STOP_WORDS_FOLDER_PATH)
    pos_score, neg_score, pol_score, subj_score = calculate_scores(cleaned_text)
    overall_sentiment = "POSITIVE🙂" if pol_score > 0 else "NEGATIVE☹️" if pol_score < 0 else "NEUTRAL😐"

    result = {
        "content": cleaned_text,
        "positive_score": pos_score,
        "negative_score": neg_score,
        "polarity_score": pol_score,
        "subjectivity_score": subj_score,
        "sentiment": overall_sentiment,
    }

    return render_template('result.html', result=result)

if __name__ == "__main__":
    app.run(debug=True)

#import neccessary library
import pandas as pd
import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import string
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
import re
import glob
from nltk.corpus import cmudict
import time



# Ensure required NLTK data is downloaded
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('cmudict')

# Commented out IPython magic to ensure Python compatibility.
# Define file paths (using Pathlib for better cross-platform compatibility)
REQUIRENT_FOLDER_PATH = Path(__file__).parent.parent / "Blackcofee assignment"
INPUT_EXCEL_FILE = REQUIRENT_FOLDER_PATH / 'Input.xlsx'
STOP_WORDS_FOLDER_PATH = REQUIRENT_FOLDER_PATH / 'StopWords'
MASTER_DIC_FOLDER_PATH = REQUIRENT_FOLDER_PATH / 'MasterDictionary'
EXTRACTED_ARTICLES_FOLDER = REQUIRENT_FOLDER_PATH / 'extracted_articles'
OUTPUT_FILE = REQUIRENT_FOLDER_PATH / "Output.xlsx"

#outputcolumns
OUTPUT_COLUMNS = [
    'URL_ID', 'URL', 'POSITIVE SCORE', 'NEGATIVE SCORE', 'POLARITY SCORE',
    'SUBJECTIVITY SCORE', 'AVG SENTENCE LENGTH', 'PERCENTAGE OF COMPLEX WORDS',
    'FOG INDEX','AVG NUMBER OF WORDS PER SENTENCE','COMPLEX WORD COUNT','WORD COUNT',
    'SYLLABLE PER WORD','PERSONAL PRONOUNS' ,'AVG WORD LENGTH']

#load df
def get_input_as_dataframe():
    """
    Read the input Excel file into a pandas DataFrame.
    """
    if not INPUT_EXCEL_FILE.is_file():
        print(f'Input Excel file not found: {INPUT_EXCEL_FILE}')
        return None
    else:
        df = pd.read_excel(INPUT_EXCEL_FILE, engine='openpyxl')
        print('Dataframe created successfully')
        return df

def remove_common_section(text):
    """Removes common repetitive sections from the text."""
    # Combine multiple removal patterns into fewer `re.sub` calls for efficiency
    patterns = [
       
        r'Project Snapshots.*?(?=Summarize|Contact Details)',  # Remove everything after 'Project Snapshots'
        
        r'Contact Details.*',  # Remove everything after 'Contact Details'
        r'Project website url\s*https?://\S+',  # Remove 'Project website URL'
        r'Summarize.*?This project was done by the Blackcoffer Team, a Global IT Consulting firm\.',  # Remove 'Summarize' section
        r'For project discussions and daily updates, would you like to use Slack, Skype, Telegram, or Whatsapp\? Please recommend, what would work best for you\.' , # Remove specific discussion text
        r'https?://\S+|www\.\S+'
    ]
    for pattern in patterns:
        text = re.sub(pattern, '', text, flags=re.DOTALL)

    # Clean up any extra spaces
    return text.strip()

##Extract url and url_id from input file
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
    words = word_tokenize(text_without_urls.lower())
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

def calculate_scores(text, positive_words, negative_words):

    words = re.findall(r'\b\w+\b', text)
    positive_score = sum(1 for word in words if word in positive_words)
    negative_score = sum(1 for word in words if word in negative_words)
    polarity_score = (positive_score - negative_score) / ((positive_score + negative_score) + 0.000001)
    subjectivity_score = (positive_score + negative_score) / (len(words) + 0.000001)

    return positive_score, negative_score, polarity_score, subjectivity_score


##Analysis of Readability
def analyze_readability(text):
    if not isinstance(text, str) or not text.strip():
        return 0, 0, 0

    sentences = sent_tokenize(text)
    words = re.findall(r'\b\w+\b', text)
    num_sentences = len(sentences)
    num_words = len(words)

    if num_sentences == 0 or num_words == 0:
        return 0, 0, 0

    avg_sentence_length = num_words / num_sentences

    def is_complex(word):
        vowels = "aeiouy"
        syllable_count = 0
        for letter in word.lower():
            if letter in vowels:
                syllable_count += 1
        return syllable_count > 2

    complex_words = [word for word in words if is_complex(word)]
    num_complex_words = len(complex_words)
    percentage_complex_words = (num_complex_words / num_words) * 100
    fog_index = 0.4 * (avg_sentence_length + percentage_complex_words)
    return avg_sentence_length, percentage_complex_words, fog_index



##Average Number of Words Per Sentence
def average_words_per_sentence(text):

    if not isinstance(text, str) or not text.strip():
        return 0

    # Split text into sentences using common sentence-ending punctuations
    sentences = sent_tokenize(text)

    # Remove empty strings from the list of sentences
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    if not sentences:  # If no sentences exist
        return 0

    # Count the total number of words
    words = re.findall(r'\b\w+\b', text)
    total_words = len(words)

    # Calculate the average number of words per sentence
    average_words = total_words / len(sentences)
    return round(average_words, 2)

## Complex Word Count
def count_complex_words(text):
    words = re.findall(r'\b\w+\b', text)
    if not words: #Handle no words case
        return 0

    def is_complex(word):
        vowels = "aeiouy"
        syllable_count = 0
        for letter in word:
            if letter in vowels:
                syllable_count += 1
        return syllable_count > 2

    complex_word_count = sum(1 for word in words if is_complex(word))
    return complex_word_count


##Word Count

def count_cleaned_words(text):

    stop_words = set(stopwords.words('english'))
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Tokenize text into words
    words = word_tokenize(text)
    # Remove stopwords and punctuation
    cleaned_words = [
        word for word in words
        if word not in stop_words and word not in string.punctuation
    ]

    # Return the total number of cleaned words
    return len(cleaned_words)


##Syllable Count Per Word
def count_syllables(word):
    
    word = re.sub(r'[^a-z]', '', word)  # Remove non-alphabetic characters
    if not word:
        return 0
    vowels = "aeiou"
    if word.endswith("es") or word.endswith("ed"):
        word = word[:-2]
    syllable_count = sum(1 for char in word if char in vowels)
    return max(syllable_count, 1)

# Function to calculate total syllable count for a text
def calculate_syllables(text):
    words = re.findall(r'\b\w+\b', text)  # Split text into words
    return sum(count_syllables(word) for word in words)


##Personal Pronouns
def count_personal_pronouns(text):

    if not isinstance(text, str):
        return None

    pronoun_pattern = r"\b(i|we|my|ours|us)\b"  # \b ensures word boundaries
    matches = re.findall(pronoun_pattern, text.lower())

    #Exclude US
    pronouns = [match for match in matches if match != "us"]

    return len(pronouns)


#calculate avg word length 
def average_word_length(text):
    if not isinstance(text, str) or not text.strip():
        return None
    words = re.findall(r'\b\w+\b', text)
    if not words:
        return 0
    total_characters = sum(len(word) for word in words)
    avg_word= total_characters/ len(words)
    return avg_word
#write main file
def main():
    df = get_input_as_dataframe()
    if df is None:
        return

    #EXTRACTED_ARTICLES_FOLDER.mkdir(exist_ok=True)
    EXTRACTED_ARTICLES_FOLDER.mkdir(parents=True, exist_ok=True)
    result = create_positive_negative_dicts(MASTER_DIC_FOLDER_PATH, STOP_WORDS_FOLDER_PATH)

    # Load positive and negative words
    positive_words, negative_words = set(), set()
    for file in Path(MASTER_DIC_FOLDER_PATH).glob("*.txt"): 
        with open(file, 'r', encoding='latin-1') as f:
            if "positive" in file.name:
                positive_words.update(line.strip().lower() for line in f)
            elif "negative" in file.name:
                negative_words.update(line.strip().lower() for line in f)

    output_data = []

    for _, row in df.iterrows():
        url_id, url = row['URL_ID'], row['URL']
        combined_content = extract_article_title_and_content(url, url_id, EXTRACTED_ARTICLES_FOLDER)
        
        
        if not combined_content:
            continue

        
        
        cleaned_text = clean_with_stopwords(combined_content, STOP_WORDS_FOLDER_PATH)
        pos_score, neg_score, pol_score, subj_score = calculate_scores(cleaned_text, positive_words, negative_words)
        avg_sent_len, perc_complex_words, fog_index = analyze_readability(cleaned_text)
        avg_word_per_sent=average_words_per_sentence(cleaned_text)
        cnt_cmplx_word=count_complex_words(cleaned_text)
        clean_cont_words= count_cleaned_words(cleaned_text)
        syllables_cnt=calculate_syllables(cleaned_text)
        cnt_persnl_prounc=count_personal_pronouns(cleaned_text)
        avg_word_len=average_word_length(cleaned_text)

        output_data.append([url_id, url, pos_score,
                             neg_score, pol_score, 
                            subj_score,avg_sent_len, 
                            perc_complex_words, fog_index,
                            avg_word_per_sent,cnt_cmplx_word,
                            clean_cont_words,syllables_cnt,cnt_persnl_prounc,avg_word_len
                            ])

    output_df = pd.DataFrame(output_data, columns=OUTPUT_COLUMNS)  
    output_df.to_excel(OUTPUT_FILE, index=False)
    print(f"Analysis complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()

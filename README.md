# Sentiment Analysis of Blogs

This project Extract data from BLACKCOFFER blog posts using BeautifulSoup and performs sentiment analysis and calculates various text metrics on a collection of text data. It uses natural language processing techniques to analyze the sentiment of the text and measure its complexity and subjectivity. The project utilizes the Python programming language and various libraries such as Pandas, NumPy, NLTK, BeautifulSoup. 📊📝

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Results](#results)

## Installation

1. Clone the repository: `git clone https://github.com/your-username/project-name.git`
2. Navigate to the project directory: `cd project-name`
3. Install the required dependencies: `pip install pandas numpy nltk BeautifulSoup` 💻📦

## Usage

1. Prepare the data:
   - Ensure that the required data files are available in the specified directory:
     - Stop words: `Data/StopWords`
     - Master dictionary: `Data/MasterDictionary`
     - Extracted data: `Data/extracted_articles`
     - Output data structure: `Output Data Structure.xlsx`
   - **Update the data paths in the code if necessary.**

2. Execute the Python script: `Text_analysis_blackcofee_assignment.ipynb` ▶️ python file Text_analysis.py

3. Wait for the script to complete the sentiment analysis and text metrics calculation.

4. Retrieve the results:
   - The results are saved in the file `app.py`.

5. If you want to Analyse Blog use app: `Text_analysis_blackcofee_assignment
## Features

- Automatically Scraps Important data from the blog post ignoring the irrelevant information.
- Generates a list of stopwords to remove common words from the text. 🛑
- Creates a master dictionary of positive and negative words for sentiment analysis. 📚
- Retrieves content from scrapped data files based on URL IDs. 📂
- Performs sentiment analysis using the master dictionary and calculates positive and negative scores. 😃😔
- Calculates various text metrics:
  - Average number of syllables per word
  - Count and percentage of complex words
  - FOG index
  - Average sentence length
  - Average number of words per sentence
  - Word count
  - Average word length
  - Count of personal pronouns
- Outputs the results to an Excel file. 📊📝

## Results

The results of the sentiment analysis and text metrics calculation are saved in the `Output.xlsx` file. The Excel file contains a sheet with the following columns:

- URL_ID: The identifier of the URL. 🔗
- POSITIVE SCORE: The count of positive words in the text. 👍
- NEGATIVE SCORE: The count of negative words in the text. 👎
- POLARITY SCORE: The sentiment polarity score calculated as (positive score - negative score) / (positive score + negative score + 0.000001). 📈📉
- SUBJECTIVITY SCORE: The subjectivity score calculated as (positive score + negative score) / (word count + 0.000001). 📖
- AVG SENTENCE LENGTH: The average number of words per sentence. 📊
- PERCENTAGE OF COMPLEX WORDS: The percentage of complex words in the text. 📈
- FOG INDEX: The FOG index calculated as 0.4 * (average syllables per word + percentage of complex words). 🌧️
- AVG NUMBER OF WORDS PER SENTENCE: The average number of words per sentence. 📊
- COMPLEX WORD COUNT: The count of complex words in the text. 📈
- WORD COUNT: The total number of words in the text. 📊
- SYLLABLE PER WORD: The average number of syllables per word. 📏
- PERSONAL PRONOUNS: The count of personal pronouns in the text. 👥
- AVG WORD LENGTH: The average length of words in the text. 📏


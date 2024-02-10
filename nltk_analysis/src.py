"""__doc__
Sentiment Analysis Using NLTK and TextBlob

This script incorporates NLTK and TextBlob libraries to perform sentiment analysis on a set of reviews.
The code is adapted and extended from an existing project using the NLTK library available at:
https://github.com/j-balkovec/Projects/tree/main/NLP%20Bot

The script provides functionality to:
1. Strip special characters and punctuation from sentences.
2. Tokenize, stem, and preprocess sentences for sentiment analysis.
3. Calculate the polarity and subjectivity scores of sentences.
4. Categorize polarity and subjectivity into descriptive labels.
5. Dump the sentiment analysis results to a JSON file.

Dependencies:
- nltk
- textblob

Author: [Your Name]
Date: [Current Date]

Usage:
1. Ensure that the necessary dependencies are installed.
2. Modify the constants in the "__constants__" section, especially the 'DATA_SET_PATH' and 'DATA_OUT_PATH'.
3. Run the script to perform sentiment analysis on a dataset and generate results.

For detailed information about each function, refer to the function-specific documentation provided below.

__note__
- Reusing the source code/backend of project that uses the nltk library to get the polarity of a sentence.
- The code can be found here: https://github.com/j-balkovec/Projects/tree/main/NLP%20Bot
- Added functionality to fit it for the current project.
"""

"""__imports__"""


"""__constants__"""
import string
import re
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
import json
from tqdm import tqdm

from ..utility.fetch import find_file_path
import logging

LOGGER_NAME: str = "File: imbdModel_src.py"
DATA_SET_PATH: str = find_file_path("predictions.json")
DATA_OUT_PATH: str = find_file_path("nltk_analysis.json")

"""__logger__"""
logger = logging.getLogger(LOGGER_NAME)
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(find_file_path("nltk.log"))
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


def strip_characters(sentence: str) -> str:
    """__doc__
    Remove special characters, punctuation, and unnecessary whitespaces from a sentence.
    """
    sentence = re.sub(r"[^a-zA-Z0-9\s]", "", sentence)
    sentence = sentence.translate(str.maketrans("", "", string.punctuation))
    sentence = " ".join(sentence.split())

    """_adjustment_
  - Remove <UNK> tokens from the sentence.
  """
    sentence = sentence.replace("<UNK>", "")

    logger.info(f"[sentence stripped of special characters]: {sentence}")
    return sentence.lower()


def tokenize(sentence: str) -> list:
    """__doc__
    Tokenize a sentence into a list of words.
    """
    placeholder = nltk.word_tokenize(sentence)
    logger.info(f"[sentence tokenized]: {placeholder}")
    return placeholder


def stem(sentence: list) -> list:
    """__doc__
    Stem a sentence.
    """
    stemmer = nltk.stem.PorterStemmer()
    placeholder = [stemmer.stem(word) for word in sentence]
    logger.info(f"[sentence stemmed]: {placeholder}")
    return placeholder


def preprocess(sentence: str) -> list:
    """__doc__
    Preprocess a sentence.
    """
    sentence = strip_characters(sentence)
    sentence = tokenize(sentence)
    sentence = stem(sentence)

    logger.info(f"[sentence preprocessed]: {sentence}")
    return sentence


def get_polarity(sentence: list) -> float:
    """__doc__
    Calculate the polarity score of a given sentence.

    Args:
      sentence (str): The input sentence for sentiment analysis.

    Returns:
      float: The polarity score of the sentence.
    """
    sentence: str = " ".join(sentence)

    sid = SentimentIntensityAnalyzer()
    sentiment_scores = sid.polarity_scores(sentence)

    logger.info(f"[sentence polarity score calculated]: {sentiment_scores['compound']}")
    return sentiment_scores["compound"]


def get_subjectivity(sentence: list) -> float:
    """__doc__
    Calculate the subjectivity of a given sentence.

    Parameters:
    sentence (str): The input sentence for which subjectivity needs to be calculated.

    Returns:
    float: The subjectivity score of the sentence, ranging from 0.0 to 1.0.
      A score closer to 0.0 indicates objective content, while a score closer to 1.0 indicates subjective content.
    """
    sentence: str = " ".join(sentence)
    blob = TextBlob(sentence)
    logger.info(
        f"[sentence subjectivity score calculated]: {blob.sentiment.subjectivity}"
    )
    return blob.sentiment.subjectivity


def get_polarity_category(polarity: float) -> str:
    """__doc__
    Returns the polarity category based on the given polarity value.

    Parameters:
    polarity (float): The polarity value to categorize.

    Returns:
    str: The polarity category.

    """
    if polarity > 0.5:
        return "Very Positive"
    elif polarity > 0.0:
        return "Positive"
    elif polarity == 0.0:
        return "Neutral"
    elif polarity >= -0.5:
        return "Slightly Negative"
    else:
        return "Very Negative"


def get_subjectivity_category(subjectivity: float) -> str:
    """__doc__
    Determines the category of subjectivity based on the given subjectivity score.

    Args:
      subjectivity (float): The subjectivity score ranging from 0 to 1.

    Returns:
      str: The category of subjectivity, either "Subjective" or "Objective".
    """
    if subjectivity >= 0.5:
        return "Subjective"
    else:
        return "Objective"


def get_sentiment(sentence: str) -> dict:
    """__doc__
    Calculate the sentiment of a given sentence.

    Args:
      sentence (str): The input sentence for sentiment analysis.

    Returns:
      dict: A dictionary containing the sentiment analysis results, including:
        - sentence: The input sentence.
        - polarity_score: The polarity score of the sentence.
        - subjectivity_score: The subjectivity score of the sentence.
        - polarity: The polarity category of the sentence.
        - subjectivity: The subjectivity category of the sentence.
    """
    processed_sentence = preprocess(sentence)
    polarity = get_polarity(processed_sentence)
    subjectivity = get_subjectivity(processed_sentence)

    return {
        "Sentiment": {
            "sentence": sentence,
            "polarity_score": polarity,
            "subjectivity_score": subjectivity,
            "polarity": get_polarity_category(polarity),
            "subjectivity": get_subjectivity_category(subjectivity),
        }
    }


def dump_to_json_file(results: dict) -> None:
    """__doc__
    Dump the given results to a JSON file.

    Args:
      results (dict): The results to be dumped.
      file_path (str): The file path to dump the results to.
    """
    try:
        with open(DATA_OUT_PATH, "w") as json_file:
            json.dump(results, json_file, indent=4)
        print("\n-- [SUCCESS] Results written to file:", DATA_OUT_PATH, "\n")
    except Exception as e:
        print("\n-- [ERROR] Failed to write results to file:", DATA_OUT_PATH, "\n")
        print("Error:", str(e))


"""_adjustment_
- Added functionality to read the predictions.json file and print the results.
"""


def verify_sentiment():
    """__doc__
    Verify sentiment of reviews in a dataset.

    This function reads a JSON dataset file, extracts the review text from each entry,
    and calculates the sentiment score for each review using the `get_sentiment` function.
    The sentiment scores are then stored in a list and dumped to a JSON output file.

    Raises:
      FileNotFoundError: If the dataset file specified by `DATA_SET_PATH` is not found.
      json.JSONDecodeError: If there is an error decoding the JSON in the dataset file.

    """
    print("\n\n-- [INFO] Starting sentiment analysis\n\n")
    results = []
    try:
        with open(DATA_SET_PATH, "r") as json_file:
            data = json.load(json_file)

            for review in tqdm(
                data,
                desc="Processing Reviews",
                unit="reviews",
                dynamic_ncols=True,
                colour="green",
            ):
                review_text = review.get("ReviewText", "")
                results.append(get_sentiment(strip_characters(review_text)))

    except FileNotFoundError:
        print(f"File not found: {DATA_SET_PATH} \n")
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {DATA_SET_PATH} \n")

    dump_to_json_file(results)

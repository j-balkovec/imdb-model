"""
IMDB Sentiment Analysis Model

This script demonstrates building, training, and evaluating a sentiment analysis model using the IMDB dataset.
The model is a simple neural network with an embedding layer, flattening layer, dense layer, and a sigmoid output layer.
The trained model is saved, and predictions are made on a set of sample reviews.

Dependencies:
- tensorflow
- numpy
- fetch.py
- tqdm

Author: Jakob Balkovec
Date: 25th Dec 2023

Usage:
1. Make sure to update the file paths in the "__constants__" section.
2. Run the script to load the IMDB dataset, preprocess the data, build and train the model, and make predictions.

For detailed information about each function, refer to the function-specific documentation provided below.
"""

"""_links_
* Due to challenges with SSL requests during the dataset download process, I opted for a manual download.
* There is no need for concern about the dataset's location. The script is designed to automatically 
  search for the dataset in the current working directory and its subdirectories. If found, 
  it returns the full file path; otherwise, it returns None.
  
* For reference, the IMDb dataset can be manually obtained from the following links:

[iMDB Data Set Link]: https://ai.stanford.edu/~amaas/data/sentiment/
[iMDB Word Index Link]: https://s3.amazonaws.com/text-datasets/imdb_word_index.json
"""

"""__imports__"""
import sys
import tensorflow as tf
from typing import Tuple
import json
import numpy as np
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import imdb
from tqdm import tqdm

from utility.fetch import find_file_path

sys.path.append('/Users/jbalkovec/Desktop/Projects/imdbModel')

"""__constants/constexpr__"""
IMDB_DATA_SET: str = find_file_path("imdb.npz")
IMDB_JSON: str = find_file_path("index_word_index.json")
MODEL_NAME: str = find_file_path("imdb_model.keras")
OUT_JSON_FILE: str = find_file_path("predictions.json")

"""__defaults__
epochs: 5
max_seq_len: 250

input_dim: 10000
output_dim: 16

num_neurons_dense: 16
num_neurons_out: 1
"""

"""__train_param"""
EPOCHS: int = 5
MAX_SEQ_LEN: int = 250

INPUT_DIM: int = 20000
OUTPUT_DIM: int = 25

NUM_NEURONS_DENSE: int = 25
NUM_NEURONS_OUT: int = 1  # Binary Classification

NUM_WORDS: int = 10000


def load_imdb_dataset(
    path_to_file: str, num_words: int = NUM_WORDS
) -> Tuple[Tuple[list, list], Tuple[list, list]]:
    """__doc__
    Load the IMDB dataset from a file.

    Args:
      path_to_file (str): The path to the IMDB dataset file.
      num_words (int, optional): The maximum number of words to keep based on word frequency. Defaults to 10000.

    Returns:
      tuple: A tuple containing two tuples. The first tuple contains the training data and labels, and the second tuple contains the test data and labels.
    """
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(
        path=path_to_file, num_words=num_words
    )

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    test_data = np.array(test_data)
    test_labels = np.array(test_labels)

    train_labels = np.array(train_labels, dtype=np.float32)

    return (train_data, train_labels), (test_data, test_labels)


def display_dataset_info(train_data):
    """__doc__
    Display information about the dataset.

    Args:
      train_data (tuple): A tuple containing the training data, where the first element is a list of reviews and the second element is a list of labels.
    """
    print("\n[Length of the first training review]:", len(train_data[0]))
    print("[Sample Review]:", train_data[0][0])
    print("[Label]:", train_data[1][0], "\n\n")


def preprocess_sequences(
    train_data, test_data, maxlen=MAX_SEQ_LEN
) -> Tuple[np.ndarray, np.ndarray]:
    """__doc__
    Preprocesses the input sequences by padding or truncating them to a fixed length.

    Args:
      train_data (list): The training data sequences.
      test_data (list): The testing data sequences.
      maxlen (int, optional): The maximum length of the sequences. Defaults to 250.

    Returns:
      tuple: A tuple containing the preprocessed training data and testing data.
           The sequences are padded or truncated to the specified maximum length.
    """
    train_data = tf.keras.preprocessing.sequence.pad_sequences(
        train_data, maxlen=maxlen, padding="post", truncating="post"
    )
    test_data = tf.keras.preprocessing.sequence.pad_sequences(
        test_data, maxlen=maxlen, padding="post", truncating="post"
    )
    return train_data, test_data


def build_model() -> models.Sequential:
    """__doc__
    Build and return a Sequential model for sentiment analysis.

    Returns:
      models.Sequential: The built Sequential model.
    """
    model = models.Sequential()
    model.add(
        layers.Embedding(
            input_dim=INPUT_DIM, output_dim=OUTPUT_DIM, input_length=MAX_SEQ_LEN
        )
    )
    model.add(layers.Flatten())
    model.add(layers.Dense(NUM_NEURONS_DENSE, activation="relu"))
    model.add(layers.Dense(NUM_NEURONS_OUT, activation="sigmoid"))
    return model


def compile_and_train_model(
    model, train_data, train_labels, epochs, validation_data=None
) -> tf.keras.callbacks.History:
    """__doc__
    Compiles and trains the given model using the provided training data and labels.

    Args:
      model (tf.keras.Model): The model to be compiled and trained.
      train_data (numpy.ndarray): The training data.
      train_labels (numpy.ndarray): The training labels.
      epochs (int): The number of epochs to train the model.
      validation_data (tuple, optional): The validation data and labels. Defaults to None.

    Returns:
      tf.keras.callbacks.History: The training history.
    """
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    history = model.fit(
        train_data, train_labels, epochs=epochs, validation_data=validation_data
    )
    return history


def evaluate_model(model, test_data, test_labels):
    """__doc__
    Evaluates the given model on the test data and labels.

    Args:
      model: The trained model to be evaluated.
      test_data: The input test data.
      test_labels: The corresponding test labels.

    Returns:
      None
    """
    test_loss, test_acc = model.evaluate(test_data, test_labels)
    print(f"\n\n[Test accuracy]: {test_acc} \n[Test loss]: {test_loss}\n\n")


def decode_review(encoded_review) -> str:
    """__doc__
    Decodes the encoded review using the word index dictionary.

    Parameters:
    encoded_review (list): The encoded review to be decoded.

    Returns:
    str: The decoded review.
    """
    word_index = imdb.get_word_index(path=IMDB_JSON)
    reverse_word_index = dict([(value + 3, key) for (key, value) in word_index.items()])
    reverse_word_index[0] = "<PAD>"
    reverse_word_index[1] = "<START>"
    reverse_word_index[2] = "<UNK>"
    decoded_review = " ".join(
        [reverse_word_index.get(i, "<UNK>") for i in encoded_review]
    )
    decoded_review = decoded_review.strip("<PAD> <START> <UNK>")
    return decoded_review


def write_predictions_to_file(results: str) -> bool:
    """__doc__
    Writes the predictions to a file.

    Returns:
      bool: True if the predictions were written to a file, False otherwise.
    """
    try:
        with open(OUT_JSON_FILE, "w", encoding="utf-8") as f:
            f.write(results)
            print(f"\n\n[SUCCESS] predictions written to file: {OUT_JSON_FILE}\n\n")
        return True
    except Exception as e:
        print(f"[ERROR] failed writing predictions to file: {e}")
        return False


def make_predictions(model, sample_reviews, test_labels):
    """__doc__
    Makes predictions using the given model on a set of sample reviews.

    Args:
      model (object): The trained model used for prediction.
      sample_reviews (list): List of encoded sample reviews.
      test_labels (list): List of corresponding labels for the sample reviews.
    """
    predictions = model.predict(sample_reviews)

    results = []

    print("\n------------------------------------\n")
    for i in tqdm(
        range(len(sample_reviews)),
        desc="[Making Predictions]",
        unit=" reviews",
        dynamic_ncols=True,
        colour="green",
    ):
        review_result = {
            "review_number": i + 1,
            "actual_sentiment": "very positive"
            if test_labels[i] == 2
            else "positive"
            if test_labels[i] == 1
            else "neutral"
            if test_labels[i] == 0
            else "negative"
            if test_labels[i] == -1
            else "very negative",
        }

        polarity = predictions[i][0]
        if polarity > 0.5:
            predicted_sentiment = "very positive"
        elif polarity > 0.0:
            predicted_sentiment = "positive"
        elif polarity == 0.0:
            predicted_sentiment = "neutral"
        elif polarity >= -0.5:
            predicted_sentiment = "slightly negative"
        else:
            predicted_sentiment = "very negative"

        review_result["predicted_sentiment"] = predicted_sentiment
        review_result["review_text"] = decode_review(sample_reviews[i])

        results.append(review_result)

    results_json = json.dumps(results, indent=4)
    write_predictions_to_file(results=results_json)


def save_model(model, model_name):
    """__doc__
    Save the trained model to a file.

    Args:
      model (object): The trained model object to be saved.
      model_name (str): The name of the file to save the model to.
    """
    model.save(model_name)


def train_model(
    model, train_data, train_labels, epochs, validation_data=None
) -> tf.keras.callbacks.History:
    """__doc__
    Trains the given model on the provided training data and labels for the specified number of epochs.

    Args:
      model (object): The model to be trained.
      train_data (numpy.ndarray): The training data.
      train_labels (numpy.ndarray): The training labels.
      epochs (int): The number of epochs to train the model.
      validation_data (tuple, optional): Validation data and labels. Defaults to None.

    Returns:
      tf.keras.callbacks.History: The training history.

    """
    history = compile_and_train_model(
        model, train_data, train_labels, epochs=epochs, validation_data=validation_data
    )
    save_model(model, MODEL_NAME)
    return history


def run_model(model, sample_reviews, test_labels):
    """__doc__
    Runs the given model on the sample reviews and test labels.

    Args:
      model: The model to be used for making predictions.
      sample_reviews: The sample reviews to be used for prediction.
      test_labels: The corresponding labels for the sample reviews.
    """
    make_predictions(model, sample_reviews, test_labels)
    save_model(model, MODEL_NAME)


def main() -> None:
    """__doc__
    Main function to train, evaluate, and make predictions using the IMDb model.

    Returns:
      None
    """
    (train_data, train_labels), (test_data, test_labels) = load_imdb_dataset(
        IMDB_DATA_SET
    )
    display_dataset_info(train_data)
    train_data, test_data = preprocess_sequences(train_data, test_data)
    model = build_model()
    train_model(
        model,
        train_data,
        train_labels,
        epochs=EPOCHS,
        validation_data=(test_data, test_labels),
    )
    evaluate_model(model, test_data, test_labels)
    make_predictions(model, test_data, test_labels)


if __name__ == "__main__":
    main()

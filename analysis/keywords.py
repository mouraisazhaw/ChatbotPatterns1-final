import sqlite3
from collections import Counter
import re
from nltk.corpus import stopwords
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# Download stopwords and vader lexikon
nltk.download('stopwords')
nltk.download('vader_lexicon')

# List of stopwords from nltk
STOPWORDS = set(stopwords.words('english'))

def get_top_words_by_escalation(db_path, top_n=20):
    """
    Extracts the most frequent words for each escalation level from the database,
    ignoring stopwords.

    :param db_path: Path to the SQLite database.
    :param top_n: Number of top words to extract.
    :return: A dictionary with escalation levels as keys and top words as values.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve messages and escalation levels
    query = "SELECT escalation, message FROM conversations WHERE message IS NOT NULL"
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()

    # Group data by escalation levels
    escalation_groups = {}
    for escalation, message in data:
        if escalation not in escalation_groups:
            escalation_groups[escalation] = []
        escalation_groups[escalation].append(message)

    # Function to tokenize and calculate word frequency
    def get_top_words(texts):
        all_words = []
        for text in texts:
            # Tokenize: Extract only words (no special characters)
            words = re.findall(r'\b\w+\b', text.lower())
            # Remove stopwords
            filtered_words = [word for word in words if word not in STOPWORDS]
            all_words.extend(filtered_words)
        return Counter(all_words).most_common(top_n)

    # Calculate top words for each escalation level
    results = {}
    for escalation, messages in escalation_groups.items():
        results[escalation] = get_top_words(messages)

    return results

def extract_negative_words_vader(messages):
    """
    Extracts negative words from a list of messages using VADER.

    :param messages: List of text messages.
    :return: List of negative words.
    """
    sia = SentimentIntensityAnalyzer()
    negative_words = []

    for message in messages:
        # Tokenize: Extract words
        words = re.findall(r'\b\w+\b', message.lower())
        for word in words:
            # Calculate sentiment score for each word
            score = sia.polarity_scores(word)
            if score['compound'] < 0:  # Negative word
                negative_words.append(word)

    return Counter(negative_words).most_common(20)

def extract_negative_words_by_escalation(db_path, top_n=20):
    """
    Extracts the most frequent negative words for each escalation level from the database.

    :param db_path: Path to the SQLite database.
    :param top_n: Number of top negative words to extract.
    :return: A dictionary with escalation levels as keys and top words as values.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Retrieve messages and escalation levels
    query = "SELECT escalation, message FROM conversations WHERE message IS NOT NULL"
    cursor.execute(query)
    data = cursor.fetchall()
    conn.close()

    # Group data by escalation levels
    escalation_groups = {}
    for escalation, message in data:
        if escalation not in escalation_groups:
            escalation_groups[escalation] = []
        escalation_groups[escalation].append(message)

    # Initialize VADER sentiment analyzer
    sia = SentimentIntensityAnalyzer()

    # Function to extract negative words
    def get_negative_words(messages):
        negative_words = []
        for message in messages:
            # Tokenize: Extract words
            words = re.findall(r'\b\w+\b', message.lower())
            for word in words:
                # Calculate sentiment score for each word
                score = sia.polarity_scores(word)
                if score['compound'] < 0:  # Negative word
                    negative_words.append(word)
        return Counter(negative_words).most_common(top_n)

    # Calculate negative words for each escalation level
    results = {}
    for escalation, messages in escalation_groups.items():
        results[escalation] = get_negative_words(messages)

    return results

# Example usage
if __name__ == "__main__":
    db_path = "chatbot_conversations.db"  # Path to the database
    top_words = get_top_words_by_escalation(db_path)

    # Print results
    for escalation, words in top_words.items():
        print(f"Top words for escalation level '{escalation}':")
        for word, count in words:
            print(f"{word}: {count}")
        print("\n")

    # Example usage for negative words
    messages = ["This is terrible and broken.", "I am very upset with the service.", "Everything is fine."]
    negative_words = extract_negative_words_vader(messages)
    print("Negative words:", negative_words)

    # Example usage for negative words by escalation level
    negative_words_by_escalation = extract_negative_words_by_escalation(db_path)

    # Print results
    for escalation, words in negative_words_by_escalation.items():
        print(f"Negative words for escalation level '{escalation}':")
        for word, count in words:
            print(f"{word}: {count}")
        print("\n")
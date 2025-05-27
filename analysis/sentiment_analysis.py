import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from transformers import pipeline

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

# Initialize the RoBERTa model (Siebert)
transformer_sentiment = pipeline(
    "sentiment-analysis",
    model="siebert/sentiment-roberta-large-english"
)

# Initialize the model for toxic language
toxicity_detector = pipeline("text-classification", model="unitary/toxic-bert")

# Initialize the DistilBERT model
distilbert_sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

def analyze_transformer_sentiment(text):
    """
    Uses the Siebert RoBERTa model for sentiment analysis.
    Returns 'POSITIVE' or 'NEGATIVE'.
    """
    try:
        result = transformer_sentiment(text[:512])[0]
        label = result['label'].upper()  # Siebert returns 'POSITIVE'/'NEGATIVE'
        return label, result['score']
    except Exception as e:
        print(f"Error during transformer analysis: {e}")
        return "UNKNOWN", 0.0

def analyze_distilbert_sentiment(text):
    """
    Uses the DistilBERT model for sentiment analysis.
    Returns 'POSITIVE' or 'NEGATIVE'.
    """
    try:
        result = distilbert_sentiment(text[:512])[0]
        return {
            "label": result["label"].upper(),  # "POSITIVE" or "NEGATIVE"
            "score": result["score"]
        }
    except Exception as e:
        print(f"Error during DistilBERT analysis: {e}")
        return {
            "label": "UNKNOWN",
            "score": 0.0
        }

def analyze_toxicity(message):
    """
    Uses the 'unitary/toxic-bert' model to detect toxic content.
    Returns True if the message is toxic, otherwise False.
    """
    try:
        result = toxicity_detector(message[:512])[0]
        if result['label'] == 'TOXIC' and result['score'] > 0.8:
            return True, result['score']
        return False, result['score']
    except Exception as e:
        print(f"Error during toxicity analysis: {e}")
        return False, 0.0

def analyze_sentiments(message):
    """
    Analyzes a message using TextBlob, VADER, a transformer model, and 'unitary/toxic-bert' and evaluates the escalation level.
    """
    # TextBlob analysis
    blob = TextBlob(message)
    textblob_polarity = blob.sentiment.polarity
    textblob_subjectivity = blob.sentiment.subjectivity

    # VADER analysis
    vader_scores = sia.polarity_scores(message)
    vader_compound = vader_scores['compound']
    vader_neg = vader_scores['neg']
    vader_neu = vader_scores['neu']
    vader_pos = vader_scores['pos']

    # Transformer analysis
    transformer_label, transformer_score = analyze_transformer_sentiment(message)

    # DistilBERT analysis
    distilbert_result = analyze_distilbert_sentiment(message)
    distilbert_label = distilbert_result["label"]
    distilbert_score = distilbert_result["score"]

    # Toxicity analysis
    is_toxic, toxicity_score = analyze_toxicity(message)

    # Weighted evaluation
    score = 0
    escalation_reasons = []

    # Weighting for VADER
    if vader_compound < -0.8:
        score += 2
        escalation_reasons.append("vader_strong_negative")
    elif vader_compound < -0.5:
        score += 1
        escalation_reasons.append("vader_negative")

    # Weighting for TextBlob
    if textblob_polarity < -0.2:
        score += 1
        escalation_reasons.append("textblob_negative")
    if textblob_polarity < -0.5:
        score += 2
        escalation_reasons.append("textblob_strong_negative")

    # Weighting for Transformer
    if transformer_label == 'NEGATIVE' and transformer_score > 0.85:
        score += 2
        escalation_reasons.append("transformer_negative")
    if transformer_label == 'NEGATIVE' and transformer_score > 0.95:
        score += 3
        escalation_reasons.append("transformer_strong_negative")

    # Weighting for DistilBERT
    if distilbert_label == 'NEGATIVE':
        if distilbert_score > 0.99:
            score += 1  # previously: 3
            escalation_reasons.append("distilbert_strong_negative")
        elif distilbert_score > 0.95:
            score += 1  # previously: 3
        elif distilbert_score > 0.85:
            score += 1  # previously: 2
            escalation_reasons.append("distilbert_negative")

    # Weighting for toxicity
    if is_toxic:
        score += 3
        escalation_reasons.append("toxicity")

    # Adjusted escalation evaluation
    if score >= 5:  # Adjusted: High score -> "strong"
        escalation = "strong"
    elif score >= 2:  # Adjusted: Medium score -> "light"
        escalation = "light"
    else:
        escalation = "none"

    return {
        'textblob': {
            'polarity': textblob_polarity,
            'subjectivity': textblob_subjectivity
        },
        'vader': {
            'compound': vader_compound,
            'neg': vader_neg,
            'neu': vader_neu,
            'pos': vader_pos
        },
        'transformer': {
            'label': transformer_label,
            'score': transformer_score
        },
        'distilbert': {
            'label': distilbert_label,
            'score': distilbert_score
        },
        'toxicity': {
            'is_toxic': is_toxic,
            'score': toxicity_score
        },
        'sentiment': {
            'overall': vader_compound,
            'textblob_polarity': textblob_polarity,
            'transformer_score': transformer_score,
            'distilbert_score': distilbert_score
        },
        'escalation': escalation,
        'escalation_reason': escalation_reasons
    }
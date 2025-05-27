from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
from transformers import pipeline
from analysis.sentiment_analysis import analyze_sentiments

# Lade Sentiment-Analyse-Modell von Hugging Face (DistilBERT)
transformer_sentiment = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)

# Lade ein vortrainiertes Modell zur Erkennung von toxischer Sprache
toxicity_detector = pipeline("text-classification", model="unitary/toxic-bert")

ESCALATION_KEYWORDS = [
    "idiot", "stupid", "nonsense", "shut up", "you don't understand",
    "propaganda", "hate", "ignorant", "fool", "trash", "insult", "disgrace",
    "moron", "dumb", "worthless", "clueless", "pathetic", "loser", "garbage",
    "nobody cares", "get lost", "useless", "annoying", "ridiculous", "absurd",
    "what's wrong with you", "are you serious", "how dare you", "shame on you"
]

def contains_escalation_keywords_or_toxicity(text):
    """
    Prüft, ob der Text Eskalations-Schlüsselwörter enthält oder als toxisch eingestuft wird.
    """
    # Schlüsselworterkennung
    text = text.lower()
    for word in ESCALATION_KEYWORDS:
        if re.search(rf"\b{re.escape(word)}\b", text):
            return True

    # Toxizitätsanalyse
    try:
        result = toxicity_detector(text[:512])[0]  # Kürze bei langen Texten
        if result['label'] == 'TOXIC' and result['score'] > 0.9:
            return True
    except Exception as e:
        print(f"Fehler bei der Toxizitätsanalyse: {e}")

    return False


#
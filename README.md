# Chatbot Patterns Analysis

This project analyzes chatbot conversations to detect escalation levels, sentiment trends, and other patterns. It includes tools for generating conversations, sentiment analysis, and visualizing results.

## Key Components

### `core/chatbot2.py`
This module handles chatbot conversation generation. It supports:
- **Conversation Modes**: Neutral, provocative, and extreme.
- **Pro-Con Conversations**: Simulates debates between Pro-Bot and Con-Bot.

### `analysis/sentiment_analysis.py`
Provides sentiment analysis for chatbot messages using:
- **TextBlob**: Polarity and subjectivity.
- **VADER**: Compound sentiment scores.
- **Transformer Models**: Sentiment classification (e.g., DistilBERT, RoBERTa).


## Additional Analyses and Visualizations
- **Escalation Analysis**: Detects escalation levels by host mode or topic.
- **Pro-Con Visualization**: Compares sentiment and escalation trends between Pro-Bot and Con-Bot.
- **Dashboard**: Interactive Streamlit app for exploring conversation data.
- **Final Visualization**: Advanced clustering and feature importance analysis.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Add your OpenAI API key to `.env`.
3. Run analyses or visualizations using the provided scripts.

## License
This project is licensed under the MIT License.

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

## Main Commands

### Generate Conversations with Specific Parameters
```bash
python3 conversation_manager.py --num_conversations 1 --category "Politics" --host_mode "provocative" --api "openai"
```
- **`conversation_manager.py`**: Manages the generation of chatbot conversations.
- **`--num_conversations`**: Specifies the number of conversations to generate (e.g., `1`).
- **`--category`**: Sets the topic of the conversation (e.g., `"Politics"`).
- **`--host_mode`**: Defines the chatbot's tone or behavior (e.g., `"provocative"`).
- **`--api`**: Specifies the API to use for generating responses (e.g., `"openai"`).

### Simulate Pro-Con Conversations
```bash
python chatbot2.py --mode pro-con --api openai --category "Religion" --mode-type extreme
```
- **`chatbot2.py`**: Handles advanced conversation simulations.
- **`--mode`**: Sets the conversation type (e.g., `"pro-con"` for debates between Pro-Bot and Con-Bot).
- **`--api`**: Specifies the API to use (e.g., `"openai"`).
- **`--category`**: Defines the topic (e.g., `"Religion"`).
- **`--mode-type`**: Adjusts the intensity of the conversation (e.g., `"extreme"` for highly polarized debates).
